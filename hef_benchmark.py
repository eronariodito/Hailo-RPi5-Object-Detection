import cv2
import numpy as np
import os
import hailo_platform as hpf
from tqdm import tqdm
import time
from pathlib import Path
import json
import glob
import yaml
from colorama import Fore, Style

from utils.parser import parse_arguments_benchmark
from utils.utils import setup_logger, val_dir
from utils.pycoco_val import validate_pycoco
from utils.utils_v9 import preprocess, postprocess_v9
from utils.utils_v8 import preprocess, postprocess_v8

class HEFBenchmark:
    def __init__(self):
        self.args = parse_arguments_benchmark()
        self.output_dir = val_dir()
        self.logger = setup_logger(self.output_dir)

        self.conf_all = self.args.conf_all
        self.conf = self.args.conf
        self.iou = self.args.iou
        self.count = 0
        self.model_path = self.args.model_path
        
        params = hpf.VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
        
        self.hef = hpf.HEF(self.model_path)
        self.target = hpf.VDevice(params)
        self.infer_model = self.target.create_infer_model(self.model_path)

        configure_params = hpf.ConfigureParams.create_from_hef(self.hef, interface=hpf.HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        if "v9" in self.args.model_path:
            self.args.model_arch = "v9"
        elif "v8" in self.args.model_path:
            self.args.model_arch = "v8"
        else:
            print(f"{Fore.YELLOW}⚠️ Model version not found.{Style.RESET_ALL} {Fore.CYAN}Enter version (v8 or v9):{Style.RESET_ALL}", end=' ')
            self.args.model_arch= input().strip().lower()

        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.UINT8)
        if self.args.model_arch == "v9":
            self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.UINT8)
        elif self.args.model_arch == "v8":
            self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        self.input_shape = self.input_vstream_info.shape
        self.input_height = self.input_shape[0]
        self.input_width = self.input_shape[1]

        self.quantization_dict = {}

        for output_stream in self.output_vstream_info:
            quant_infos = output_stream.quant_info # Get first quantization info
            self.quantization_dict[output_stream.name] = [
                quant_infos.qp_scale,
                quant_infos.qp_zp
            ]

        with open(self.args.dataset, "r") as f:
            data = yaml.safe_load(f)
        
        self.classes = data.get("names", {})

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.args.dataset_path = os.getcwd() + '/' + data["path"]

        print("\n")
        folders = glob.glob(self.args.dataset_path + "/images/val*")

        if folders:
            self.val_folder = folders[0]  # Get the first match (or loop through if multiple exist)
            self.logger.info(f"📁 {Fore.MAGENTA}Found folder: {Style.RESET_ALL}{self.val_folder}")
        else:
            raise FileNotFoundError(f"No folder starting with 'val' found in {self.args.dataset_path}")
        
        folders = glob.glob(self.args.dataset_path + "/annotation*")

        if folders:
            ann_folder = folders[0]  # Get the first match (or loop through if multiple exist)
            self.logger.info(f"📁 {Fore.MAGENTA}Found annotation folder: {Style.RESET_ALL}{ann_folder}")     
                    
            files = glob.glob(f"{ann_folder}/*val*.json")

            if files:
                self.ann_file = files[0]
                self.logger.info(f"📁 {Fore.MAGENTA}Found annotation json: {Style.RESET_ALL}{self.ann_file}")

                self.pycoco = True
            else:
                self.logger.info(f"⚠️ {Fore.YELLOW}No annotation json found, skipping pycocoeval{Style.RESET_ALL}")
                self.pycoco = False
        else:
            self.logger.info(f"⚠️ {Fore.YELLOW}No annotation folder found, skipping pycocoeval{Style.RESET_ALL}")
            self.pycoco = False
        
        print("\n")

        self.coco_mapping_80to91 = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 13, "13": 14, "14": 15, "15": 16, "16": 17, "17": 18, "18": 19, "19": 20, "20": 21, "21": 22, "22": 23, "23": 24, "24": 25, "25": 27, "26": 28, "27": 31, "28": 32, "29": 33, "30": 34, "31": 35, "32": 36, "33": 37, "34": 38, "35": 39, "36": 40, "37": 41, "38": 42, "39": 43, "40": 44, "41": 46, "42": 47, "43": 48, "44": 49, "45": 50, "46": 51, "47": 52, "48": 53, "49": 54, "50": 55, "51": 56, "52": 57, "53": 58, "54": 59, "55": 60, "56": 61, "57": 62, "58": 63, "59": 64, "60": 65, "61": 67, "62": 70, "63": 72, "64": 73, "65": 74, "66": 75, "67": 76, "68": 77, "69": 78, "70": 79, "71": 80, "72": 81, "73": 82, "74": 84, "75": 85, "76": 86, "77": 87, "78": 88, "79": 89, "80": 90}
        self.time_metrics = {"preprocess": 0, "inference": 0, "postprocess": 0, "total": 0}
    
        filename = os.path.basename(self.args.model_path)
        self.logger.info(f"{Fore.CYAN}🚀 Running benchmark for{Style.RESET_ALL} {filename}")
        self.logger.info(f"{Fore.CYAN}📏 Input image size{Style.RESET_ALL} {self.input_shape}{Style.RESET_ALL} \n")


    def extract_detections_json(self, input_data: list, image: np.ndarray, threshold: float = 0.5, image_id: int = 0) -> dict:
        """
        Extract detections from the input data.

        Args:
            input_data (list): Raw detections from the model.
            threshold (float): Score threshold for filtering detections. Defaults to 0.5.

        Returns:
            dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
        """
        coco_json = []

        # Values used for scaling coords and removing padding
        img_height, img_width = image.shape[:2]
        size = max(img_height, img_width)
        padding_length = int(abs(img_height - img_width) / 2)

        for class_id, detection in enumerate(input_data):
            if len(detection) == 0:
                continue

            for det in detection:
                bbox, score = det[:4], det[4]

                if score >= threshold:
                    scaled_box = self.denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)

                    ymin, xmin, ymax, xmax = scaled_box
                    x = xmin
                    y = ymin
                    width = xmax - xmin
                    height = ymax- ymin

                    coco_json.append({
                        "image_id": int(image_id),
                        "category_id": self.coco_mapping_80to91[f"{int(class_id) + 1}"],
                        "bbox": [float(x), float(y), float(width), float(height)],
                        "score": float(score)
                    })

        return coco_json

    def run(self):
        detection_json = []
        i = 0
        plot = True

        image_path = Path(self.val_folder)
        files = [f for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']]
        
        if hasattr(self, "ann_file"):
            with open(self.ann_file, 'r') as f:
                coco_data = json.load(f)

        image_path = Path(self.val_folder)
        files = [f for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']]

        for idx,item in tqdm(enumerate(files), total=len(files), desc=f"🧠 {Fore.CYAN}Inference{Style.RESET_ALL}"):
            if hasattr(self, "ann_file"):
                image_name = os.path.basename(item)
                matching_annotations = [anno for anno in coco_data['images'] if anno['file_name'] == image_name]
                image_id = matching_annotations[0]['id']
            else:
                image_id = 1

            preprocess_tic = time.time()

            img = cv2.imread(item, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

            input_data, pad = preprocess(self, img)

            self.time_metrics["preprocess"] += time.time() - preprocess_tic

            inference_tic = time.time()
            with hpf.InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                if self.args.model_arch == "v8":
                    infer_pipeline.set_nms_iou_threshold(self.iou)
                    infer_pipeline.set_nms_score_threshold(self.conf_all)
                input_data = {self.input_vstream_info.name: np.expand_dims(input_data, axis=0)}
                results = infer_pipeline.infer(input_data)
                result = {}
                for output_stream in self.output_vstream_info:
                    result[output_stream.name] = results[output_stream.name]
                if len(self.output_vstream_info) > 1:
                    for key, value in result.items():
                        result[key] = (value.astype(np.float32) - self.quantization_dict[key][1]) * self.quantization_dict[key][0]

            self.time_metrics["inference"] += time.time() - inference_tic

            postprocess_tic = time.time()

            if self.args.model_arch == "v9":
                # indices, boxes, scores, class_ids = postprocess_v9(self, img, result, plot, image_id)
                result = postprocess_v9(self, img, result, plot, image_id)
            if self.args.model_arch == "v8":
                for key, item in result.items():
                    result = item[0]
                result = postprocess_v8(self, img, result, plot, image_id)
            
            self.time_metrics["postprocess"] += time.time() - postprocess_tic
            self.time_metrics["total"] += time.time() - preprocess_tic  

            if result is not None:
                detection_json.extend(result)
                
            if i < 10:
                plot = True
                # Filename
                filename = Path(self.output_dir) / f'output{i}_{image_id}.jpg'
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, img)     
            else:
                plot = False
            i+=1
        
        with open(Path(self.output_dir) / "prediction.json", "w") as json_file:
            json.dump(detection_json, json_file, indent=4)

        average_time = {key: (value / len(files)) * 1000 for key, value in self.time_metrics.items()}

        with open(Path(self.output_dir) / "time.json", "w") as json_file:
            json.dump(self.time_metrics, json_file, indent=4)
                 
        # Then you can log with colors and it will work correctly in both places
        self.logger.info(f"\n{Fore.CYAN}📊 Average Timing Summary{Style.RESET_ALL}")
        self.logger.info(f"{Fore.YELLOW}Preprocess:{Style.RESET_ALL}    {average_time['preprocess']:.2f} ms")
        self.logger.info(f"{Fore.YELLOW}Inference:{Style.RESET_ALL}     {average_time['inference']:.2f} ms")
        self.logger.info(f"{Fore.YELLOW}Postprocess:{Style.RESET_ALL}   {average_time['postprocess']:.2f} ms")
        self.logger.info(f"{Fore.GREEN}Total:{Style.RESET_ALL}         {average_time['total']:.2f} ms")
        self.logger.info(f"\n{Fore.MAGENTA}📁 Output saved to: {Style.RESET_ALL}{Path(self.output_dir)}")

        return detection_json

if __name__ == "__main__":
    benchmark = HEFBenchmark()

    preds_file = benchmark.run()
    if benchmark.pycoco:
        validate_pycoco(benchmark, preds_file)
