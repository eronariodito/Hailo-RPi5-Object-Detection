import cv2
import numpy as np
import os
from utils import HailoAsyncInference
import hailo_platform as hpf
from tqdm import tqdm
import time
from pathlib import Path
import json
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class HEFBenchmark:
    def __init__(self):
        self.conf_all = 0.001
        self.conf = 0.25
        self.iou = 0.7
        self.count = 0
        self.model_path = os.getcwd() + "/models/yolov8s_h8l.hef"
        
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
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.UINT8)
        self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        self.input_shape = self.input_vstream_info.shape
        output_shape = self.output_vstream_info.shape
        
        self.dataset = os.getcwd() + '/datasets/coco'
        
        print("\n")
        folders = glob.glob(self.dataset + "/images/val*")

        if folders:
            self.val_folder = folders[0]  # Get the first match (or loop through if multiple exist)
            print(f"Found folder: {self.val_folder}")
        else:
            raise FileNotFoundError(f"No folder starting with 'val' found in {self.dataset}")
        
        folders = glob.glob(self.dataset + "/annotation*")

        if folders:
            ann_folder = folders[0]  # Get the first match (or loop through if multiple exist)
            print(f"Found annotation folder: {ann_folder}")
        else:
            print("No annotation folder found, skipping pycocoeval")
            return
        
        files = glob.glob(f"{ann_folder}/*val*.json")

        if files:
            self.ann_file = files[0]
            print(f"Found annotation json: {self.ann_file}")
            self.pycoco = True
        else:
            print("No annotation json found, skipping pycocoeval")
            self.pycoco = False
        print("\n")

        with open("coco.txt", 'r', encoding="utf-8") as f:
            self.labels = f.read().splitlines()

        self.coco_mapping_80to91 = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 13, "13": 14, "14": 15, "15": 16, "16": 17, "17": 18, "18": 19, "19": 20, "20": 21, "21": 22, "22": 23, "23": 24, "24": 25, "25": 27, "26": 28, "27": 31, "28": 32, "29": 33, "30": 34, "31": 35, "32": 36, "33": 37, "34": 38, "35": 39, "36": 40, "37": 41, "38": 42, "39": 43, "40": 44, "41": 46, "42": 47, "43": 48, "44": 49, "45": 50, "46": 51, "47": 52, "48": 53, "49": 54, "50": 55, "51": 56, "52": 57, "53": 58, "54": 59, "55": 60, "56": 61, "57": 62, "58": 63, "59": 64, "60": 65, "61": 67, "62": 70, "63": 72, "64": 73, "65": 74, "66": 75, "67": 76, "68": 77, "69": 78, "70": 79, "71": 80, "72": 81, "73": 82, "74": 84, "75": 85, "76": 86, "77": 87, "78": 88, "79": 89, "80": 90}
        self.time_metrics = {"preprocess": 0, "inference": 0, "postprocess": 0, "total": 0}

        self.output_dir = self.val_dir()

        filename = os.path.basename(self.model_path)
        print(f"Running benchmark for {filename}")
        print(f"Input image size {self.input_shape} \n")

    def val_dir(self):
        base_path = os.getcwd() + "/run/val"
        base_dir = Path(base_path)
        base_dir.mkdir(parents=True, exist_ok=True)  # Creates 'run/val' if missing

        # Find existing valX folders
        existing_folders = [d for d in os.listdir(base_dir) if d.startswith("val") and d[3:].isdigit()]
        
        # Extract numbers and determine the next available one
        existing_numbers = sorted([int(d[3:]) for d in existing_folders]) if existing_folders else []
        next_number = existing_numbers[-1] + 1 if existing_numbers else 1

        # Create new directory
        new_folder = base_dir / f"val{next_number}"
        new_folder.mkdir()
        
        return new_folder

    def extract_detections(self, input_data: list, threshold: float = 0.5) -> dict:
        """
        Extract detections from the input data.

        Args:
            input_data (list): Raw detections from the model.
            threshold (float): Score threshold for filtering detections. Defaults to 0.5.

        Returns:
            dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
        """
        boxes, scores, classes = [], [], []
        num_detections = 0

        for i, detection in enumerate(input_data):
            if len(detection) == 0:
                continue

            for det in detection:
                bbox, score = det[:4], det[4]

                if score >= threshold:
                    boxes.append(bbox)
                    scores.append(score)
                    classes.append(i)
                    num_detections += 1

        return {
            'detection_boxes': boxes, 
            'detection_classes': classes, 
            'detection_scores': scores,
            'num_detections': num_detections
        }

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

    def draw_detections(self, detections: dict, image: np.ndarray, min_score: float = 0.45, scale_factor: float = 1):
        """
        Draw detections on the image.

        Args:
            detections (dict): Detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
            image (np.ndarray): Image to draw on.
            min_score (float): Minimum score threshold. Defaults to 0.45.
            scale_factor (float): Scale factor for coordinates. Defaults to 1.

        Returns:
            np.ndarray: Image with detections drawn.
        """
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']

        # Values used for scaling coords and removing padding
        img_height, img_width = image.shape[:2]
        size = max(img_height, img_width)
        padding_length = int(abs(img_height - img_width) / 2)

        for idx in range(detections['num_detections']):
            if scores[idx] >= min_score:
                np.random.seed(classes[idx])
                color = tuple(np.random.randint(0, 255, size=3).tolist())

                #scaled_box = self.denormalize_and_rm_pad(boxes[idx], size, padding_length, img_height, img_width)
                #scaled_box = boxes
                self.draw_detection(image, boxes[idx], classes[idx], scores[idx] * 100.0, color, scale_factor)

        return image

    def draw_detection(self, image: np.ndarray, box: list, cls: int, score: float, color: tuple, scale_factor: float):
        """
        Draw box and label for one detection.

        Args:
            image (np.ndarray): Image to draw on.
            box (list): Bounding box coordinates.
            cls (int): Class index.
            score (float): Detection score.
            color (tuple): Color for the bounding box.
            scale_factor (float): Scale factor for coordinates.
        """
        label = f"{self.labels[cls]}: {score:.2f}%"
        ymin, xmin, ymax, xmax = box
        ymin, xmin, ymax, xmax = int(ymin * scale_factor), int(xmin * scale_factor), int(ymax * scale_factor), int(xmax * scale_factor)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (xmin + 4, ymin + 20), font, 0.5, color, 1, cv2.LINE_AA)
                            
    def preprocess(self, frame):
        
        # frame_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        frame, pad = self.letterbox(frame, (self.input_shape[0], self.input_shape[1]))
        frame = frame.astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        return frame, pad

    def letterbox(
        self, frame, new_shape):
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image with shape (H, W, C).
            new_shape (Tuple[int, int]): Target shape (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[float, float]): Padding ratios (top/height, left/width) for coordinate adjustment.
        """
        shape = frame.shape[:2]  # Current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # Resize if needed
            frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return frame, (top / frame.shape[0], left / frame.shape[1])
    
    def denormalize_and_rm_pad(self, box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
        """
        Denormalize bounding box coordinates and remove padding.

        Args:
            box (list): Normalized bounding box coordinates.
            size (int): Size to scale the coordinates.
            padding_length (int): Length of padding to remove.
            input_height (int): Height of the input image.
            input_width (int): Width of the input image.

        Returns:
            list: Denormalized bounding box coordinates with padding removed.
        """
        for i, x in enumerate(box):
            box[i] = int(x * size)
            if (input_width != size) and (i % 2 != 0):
                box[i] -= padding_length
            if (input_height != size) and (i % 2 == 0):
                box[i] -= padding_length

        return box

    def validate_pycoco(self, preds_file):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(preds_file)

        coco_eval = COCOeval(coco_gt,coco_dt,'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def run(self):
        detection_json = []

        i = 0

        image_path = Path(self.val_folder)
        files = [f for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']]
        
        for idx,item in tqdm(enumerate(files), total=len(files), desc="Inference"):
            image_id = int(item.stem.lstrip("0"))

            img = cv2.imread(item, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

            preprocess_tic = time.time()
            input_data, pad = self.preprocess(img)
            self.time_metrics["preprocess"] += time.time() - preprocess_tic

            inference_tic = time.time()
            with hpf.InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                infer_pipeline.set_nms_iou_threshold(0.7)
                infer_pipeline.set_nms_score_threshold(0.001)
                input_data = {self.input_vstream_info.name: np.expand_dims(input_data, axis=0)}
                results = infer_pipeline.infer(input_data)
                output_data = results[self.output_vstream_info.name]
            self.time_metrics["inference"] += time.time() - inference_tic

            postprocess_tic = time.time()
            result = self.extract_detections_json(output_data[0], image=img, image_id=image_id)
            self.time_metrics["postprocess"] += time.time() - postprocess_tic
            self.time_metrics["total"] += time.time() - preprocess_tic  

            if result is not None:
                detection_json.extend(result)
                
            if i < 10:
                detections = self.extract_detections(output_data[0])

                frame_with_detections = self.draw_detections(
                            detections, img, min_score = 0.045
                        )
                        
                filename = Path(self.output_dir) / f'output{i}_{image_id}.jpg'
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, img)            

            i+=1

        with open(Path(self.output_dir) / "prediction.json", "w") as json_file:
            json.dump(detection_json, json_file, indent=4) 

        average_time = {key: (value / len(files)) * 1000 for key, value in self.time_metrics.items()}
        with open(Path(self.output_dir) / "time.json", "w") as json_file:
            json.dump(self.time_metrics, json_file, indent=4)

        print(f"Average preprocessing time: {average_time['preprocess']:.2f} ms")
        print(f"Average inference time: {average_time['inference']:.2f} ms")
        print(f"Average postprocessing time: {average_time['postprocess']:.2f} ms")
        print(f"Average total time: {average_time['total']:.2f} ms")

        print(f"Output saved to {Path(self.output_dir)}")

        return detection_json


if __name__ == "__main__":
    benchmark = HEFBenchmark()

    preds_file = benchmark.run()
    if benchmark.pycoco:
        benchmark.validate_pycoco(preds_file)
