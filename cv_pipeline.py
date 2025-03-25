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

class GstOpenCVPipeline:
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)
        self.picamera_config = None
        self.image_width = 2340
        self.image_height = 1296

        self.cam_queue = queue.Queue(maxsize=1)
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Define the sink pipeline
        sink_pipeline_str = (
            "appsrc name=opencv_src format=time is-live=true do-timestamp=true block=false "
            f'caps=video/x-raw,format=RGB,width={self.image_width},height={self.image_height} ! '
            "queue max-size-buffers=3 leaky=downstream ! "
            'videoconvert !  textoverlay name=text_overlay text=" " valignment=bottom halignment=center font-desc="Arial, 36" !'
            "fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=false signal-fps-measurements=true"
        )

        self.net_path = os.getcwd() + "/resources/yolov8s_h8l.hef"
        self.pipeline = Gst.parse_launch(sink_pipeline_str)
        self.appsrc = self.pipeline.get_by_name("opencv_src")

        # Connect to display fps-measurements
        textoverlay = self.pipeline.get_by_name("text_overlay")
        if textoverlay:
            # Font selection - professional, readable font
            textoverlay.set_property("font-desc", "Arial Bold 12")  
            
            # Text styling
            textoverlay.set_property("valignment", "top")  # Vertical alignment
            textoverlay.set_property("halignment", "left")    # Horizontal alignment
            textoverlay.set_property("line-alignment", "left")
            textoverlay.set_property("xpad", 20)              # Horizontal padding
            textoverlay.set_property("ypad", 20)              # Vertical padding
            
            # Colors and visibility
            textoverlay.set_property("color", 0xFFFFFFFF)     # White text (ARGB)
            textoverlay.set_property("outline-color", 0xFF000000)  # Black outline
            textoverlay.set_property("shaded-background", True)    # Background shading
            #textoverlay.set_property("shadow", True)          # Text shadow

        self.pipeline.get_by_name("display").connect("fps-measurements", self.on_fps_measurement)

        self.hailo_inference = HailoAsyncInference(
            self.net_path, self.input_queue, self.output_queue, 1, send_original_frame=True
        )
        self.input_height, self.input_width, _ = self.hailo_inference.get_input_shape()
        

        with open("coco.txt", 'r', encoding="utf-8") as f:
            self.labels = f.read().splitlines()
        

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        new_text = f"FPS: {fps:.2f}\nDroprate: {droprate:.2f}\nAvg FPS: {avgfps:.2f}"
        self.pipeline.get_by_name("text_overlay").set_property("text", new_text)
        #print(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True

    def picamera_thread(self):
        with Picamera2() as picam2:
            if self.picamera_config is None:
                # Default configuration
                main = {'size': (self.image_width, self.image_height), 'format': 'RGB888'}
                lores = {'size': (self.image_width, self.image_height), 'format': 'RGB888'}
                controls = {}
                config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
            else:
                config = self.picamera_config
            # Configure the camera with the created configuration
            picam2.configure(config)

            #print(f"Picamera2 configuration: width={width}, height={height}, format={format_str}")
            picam2.start()

            picam2.set_controls({"FrameRate": 120})


            print("picamera_process started")
            while True:
                frame_data = picam2.capture_array('lores')

                if frame_data is None:
                    print("Failed to capture frame.")
                    break

                frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                
                self.cam_queue.put((frame))

    def preprocess_thread(self, queue):
        while True:
            try:
                item = queue.get(timeout=5)

                if item is None:
                    break

                frame = item

                input_data, pad = self.preprocess(frame)
               
                #input_data = (input_data / self.in_scale + self.in_zero_point).astype(np.int8)

                #self.input_queue.put((input_data, pad, frame, map_info, buf))
                self.input_queue.put(([frame], [input_data]))

            except Exception as e:
                print(f"Error in preprocessing thread: {e}")
                
    def preprocess(self, frame):
        
        # frame_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        frame, pad = self.letterbox(frame, (self.input_height, self.input_width))
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
                #scaled_box = boxes
                self.draw_detection(image, scaled_box, classes[idx], scores[idx] * 100.0, color, scale_factor)

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
        font_scale = min(image.shape[0], image.shape[1]) / 1000
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
            
        cv2.putText(image, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1, cv2.LINE_AA)

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


    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()

        self.picam_thread = threading.Thread(target=self.picamera_thread, args=(), daemon=True )
        self.picam_thread.start()

        self.preprocess_thread = threading.Thread(target=self.preprocess_thread, args=(self.cam_queue,), daemon=True )
        self.preprocess_thread.start()

        self.postprocess_thread = threading.Thread(target=self.postprocess_thread, args=(self.output_queue,), daemon=True )
        self.postprocess_thread.start()

        
        self.hailo_inference.run()

        try:
            # Run the main loop

            self.loop.run()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Shutting down...")
        finally:
               
            # Clean up
            self.pipeline.set_state(Gst.State.NULL)

            self.cam_queue.put(None)
            self.input_queue.put(None)
            self.output_queue.put(None)

            if self.picam_thread.is_alive():
                self.picam_thread.join()
            if self.preprocess_thread.is_alive():
                self.preprocess_thread.join()
            if self.postprocess_thread.is_alive():
                self.postprocess_thread.join()

            self.loop.quit()
            print("Cleanup complete. Exiting...")
        
if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()

