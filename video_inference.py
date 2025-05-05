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

from utils.parser import parse_arguments_video
from utils.utils import setup_logger, vid_dir
from utils.utils_v9 import preprocess, postprocess_v9
from utils.utils_v8 import preprocess, postprocess_v8

class HEFVideoInference:
    def __init__(self):
        self.args = parse_arguments_video()
        self.output_dir = vid_dir()
        self.logger = setup_logger(self.output_dir)

        self.conf_all = self.args.conf
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


        self.coco_mapping_80to91 = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 13, "13": 14, "14": 15, "15": 16, "16": 17, "17": 18, "18": 19, "19": 20, "20": 21, "21": 22, "22": 23, "23": 24, "24": 25, "25": 27, "26": 28, "27": 31, "28": 32, "29": 33, "30": 34, "31": 35, "32": 36, "33": 37, "34": 38, "35": 39, "36": 40, "37": 41, "38": 42, "39": 43, "40": 44, "41": 46, "42": 47, "43": 48, "44": 49, "45": 50, "46": 51, "47": 52, "48": 53, "49": 54, "50": 55, "51": 56, "52": 57, "53": 58, "54": 59, "55": 60, "56": 61, "57": 62, "58": 63, "59": 64, "60": 65, "61": 67, "62": 70, "63": 72, "64": 73, "65": 74, "66": 75, "67": 76, "68": 77, "69": 78, "70": 79, "71": 80, "72": 81, "73": 82, "74": 84, "75": 85, "76": 86, "77": 87, "78": 88, "79": 89, "80": 90}
        self.time_metrics = {"preprocess": 0, "inference": 0, "postprocess": 0, "total": 0}
    
        self.filename = os.path.basename(self.args.model_path)
        self.logger.info(f"{Fore.CYAN}🚀 Running video inference for{Style.RESET_ALL} {self.filename}")
        self.logger.info(f"{Fore.CYAN}📏 Input image size{Style.RESET_ALL} {self.input_shape}{Style.RESET_ALL} \n")

    def run(self):
        cap = cv2.VideoCapture(self.args.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        #Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(Path(self.output_dir) / f'output_{os.path.splitext(self.filename)[0]}.avi', fourcc, fps, (frame_width,  frame_height), isColor=True)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video with {total_frames} frames...")

        frame_count = 0

        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Print status every 30 frames
                self.logger.info(f"📊 Processing frame {frame_count}/{total_frames}")

            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

            input_data, pad = preprocess(self, img)

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

            if self.args.model_arch == "v9":
                # indices, boxes, scores, class_ids = postprocess_v9(self, img, result, plot, image_id)
                result = postprocess_v9(self, img, result, True, None)
            if self.args.model_arch == "v8":
                for key, item in result.items():
                    result = item[0]
                result = postprocess_v8(self, img, result, True, None)
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            out.write(img)
        cap.release()
        out.release()

        self.logger.info(f"\n{Fore.MAGENTA}📁 Output saved to: {Style.RESET_ALL}{Path(self.output_dir)}")
        return

if __name__ == "__main__":
    video = HEFVideoInference()
    video.run()
