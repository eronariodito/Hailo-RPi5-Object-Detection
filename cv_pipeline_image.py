import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from picamera2 import Picamera2
import cv2
import threading
import numpy as np
import queue
import os
from utils import HailoAsyncInference
import hailo_platform as hpf
from scipy.special import softmax


class GstOpenCVPipeline:
    def __init__(self):
        self.net_path = os.getcwd() + "/resources/yolo11s_416.hef"

        # self.hailo_inference = HailoAsyncInference(
        #     self.net_path, self.input_queue, self.output_queue, 1, send_original_frame=True
        # )
        # self.input_height, self.input_width, _ = self.hailo_inference.get_input_shape()

        with open("coco.txt", 'r', encoding="utf-8") as f:
            self.labels = f.read().splitlines()
        
        params = hpf.VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
        
        self.hef = hpf.HEF(self.net_path)
        self.target = hpf.VDevice(params)
        self.infer_model = self.target.create_infer_model(self.net_path)

        configure_params = hpf.ConfigureParams.create_from_hef(self.hef, interface=hpf.HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.UINT8)
        self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        self.input_shape = self.input_vstream_info.shape
        output_shape = self.output_vstream_info.shape
        print(output_shape)

   
    def run(self, image_path = "bus.jpg"):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        input_data, pad = self.preprocess(img)

        with hpf.InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            infer_pipeline.set_nms_iou_threshold(0.98)
            infer_pipeline.set_nms_score_threshold(0.2)
            input_data = {self.input_vstream_info.name: np.expand_dims(input_data, axis=0)}
            results = infer_pipeline.infer(input_data)
            output_data = results[self.output_vstream_info.name]

        detections = self.extract_detections(output_data[0])

        frame_with_detections = self.draw_detections(
                    detections, img, min_score = 0.05
                )
                
        filename = 'savedImageCPU5_IOU.jpg'

        img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)


        cv2.imwrite(filename, img)
                            
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

                scaled_box = self.denormalize_and_rm_pad(boxes[idx], size, padding_length, img_height, img_width)
                
                self.draw_detection(image, boxes[idx], classes[idx], scores[idx] * 100.0, color, scale_factor)

        return image
    
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
        font_scale = min(image.shape[0], image.shape[1]) / 2000
        ymin, xmin, ymax, xmax = box
        ymin, xmin, ymax, xmax = int(ymin * scale_factor), int(xmin * scale_factor), int(ymax * scale_factor), int(xmax * scale_factor)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

         # Calculate the position of the label text
        label_x = xmin
        label_y = ymin - 10 if ymin - 10 > label_height else ymin + 10

        cv2.rectangle(
            image,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        a = 1 - (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255

        if a < 0.5:
            font_color = (0, 0, 0)
        else:
            font_color = (255, 255, 255)
            
        cv2.putText(image, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_DUPLEX, font_scale, font_color, 1, cv2.LINE_AA)


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

    def postprocess_thread(self, queue):
        while True:
            try:
                item = queue.get(timeout=5)

                original_frame, infer_results = item

                if len(infer_results) == 1:
                    infer_results = infer_results[0]

                detections = self.extract_detections(infer_results)

                frame_with_detections = self.draw_detections(
                    detections, original_frame, min_score = 0.25
                )

                # Create a new Gst.Buffer from the flipped frame without copying the memory
                new_buffer = Gst.Buffer.new_wrapped(original_frame.tobytes())

                self.appsrc.emit("push-buffer", new_buffer)
            except Exception as e:
                print(f"Error in postprocessing thread: {e}")


    # def run(self):
    #     self.pipeline.set_state(Gst.State.PLAYING)

    #     self.loop = GLib.MainLoop()

    #     self.picam_thread = threading.Thread(target=self.picamera_thread, args=(), daemon=True )
    #     self.picam_thread.start()

    #     self.preprocess_thread = threading.Thread(target=self.preprocess_thread, args=(self.cam_queue,), daemon=True )
    #     self.preprocess_thread.start()

    #     self.postprocess_thread = threading.Thread(target=self.postprocess_thread, args=(self.output_queue,), daemon=True )
    #     self.postprocess_thread.start()

        
    #     self.hailo_inference.run()

    #     try:
    #         # Run the main loop

    #         self.loop.run()
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt received. Shutting down...")
    #     finally:
               
    #         # Clean up
    #         self.pipeline.set_state(Gst.State.NULL)

    #         self.cam_queue.put(None)
    #         self.input_queue.put(None)
    #         self.output_queue.put(None)

    #         if self.picam_thread.is_alive():
    #             self.picam_thread.join()
    #         if self.preprocess_thread.is_alive():
    #             self.preprocess_thread.join()
    #         if self.postprocess_thread.is_alive():
    #             self.postprocess_thread.join()

    #         self.loop.quit()
    #         print("Cleanup complete. Exiting...")
        
if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()

