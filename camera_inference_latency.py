import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from picamera2 import Picamera2
import libcamera 

import cv2
import threading
import numpy as np
import queue
import os
import sys
import yaml
from colorama import Fore, Style
import traceback

from utils.utils_hailo import HailoAsyncInference
from utils.parser import parse_arguments_inference
from utils.utils_v9 import preprocess, postprocess_v9
from utils.utils_v8 import preprocess, postprocess_v8
from utils.measurement import temp_ram_measurement
from utils.utils import inference_dir

import setproctitle

class GstOpenCVPipeline:
    def __init__(self):
        self.args = parse_arguments_inference()
        setproctitle.setproctitle(f"{os.path.basename(self.args.model_path)} Inference")

        try:
            width, height = map(int, self.args.camera_res.split(','))
        except ValueError:
            parser.error("Invalid format for --camera-res. Use format WIDTH,HEIGHT (e.g., 1280,720)")

        # Initialize GStreamer
        Gst.init(None)
        self.picamera_config = None
        self.image_width = width
        self.image_height = height

        self.cam_queue = queue.Queue(maxsize=1)
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)

        self.conf = self.args.conf
        self.iou = self.args.iou
        self.conf_all = self.args.conf

        # Define the sink pipeline
        sink_pipeline_str = (
            "appsrc name=opencv_src format=time is-live=true do-timestamp=true block=false "
            f'caps=video/x-raw,format=RGB,width={self.image_width},height={self.image_height} ! '
            "queue max-size-buffers=3 leaky=downstream ! "
            'videoconvert !  textoverlay name=text_overlay text=" " valignment=bottom halignment=center font-desc="Arial, 36" !'
            "fpsdisplaysink name=display video-sink='autovideosink' sync=false text-overlay=false signal-fps-measurements=true"
        )

        self.net_path = self.args.model_path
        self.pipeline = Gst.parse_launch(sink_pipeline_str)
        self.appsrc = self.pipeline.get_by_name("opencv_src")
        self.sink = self.pipeline.get_by_name("display")
        self.sink_pad = self.sink.get_static_pad("sink")
        self.sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.sink_probe)

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

        try:
            self.hailo_inference = HailoAsyncInference(
                self.net_path, self.input_queue, self.output_queue, 1, send_original_frame=True
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1) 

        if self.args.model_arch == None:
            if "v9" in self.args.model_path:
                self.args.model_arch = "v9"
            elif "v8" in self.args.model_path:
                self.args.model_arch = "v8"
            else:
                print(f"{Fore.YELLOW}⚠️ Model version not found.{Style.RESET_ALL} {Fore.CYAN}Enter version (v8 or v9):{Style.RESET_ALL}", end=' ')
                self.args.model_arch= input().strip().lower()

        self.input_height, self.input_width, _ = self.hailo_inference.get_input_shape()
        
        with open(self.args.dataset, "r") as f:
            data = yaml.safe_load(f)

        # Load the class names from the COCO dataset
        self.classes = data.get("names", {})

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        if self.args.measure_temp_ram == True:
            self.output_dir = inference_dir()

    def measurement_temp_ram_thread(self):
        name =  os.path.splitext(os.path.basename(self.args.model_path))[0]

        output_dir = os.path.join(self.output_dir, f"{name}_measurement.csv")
        target_name = f"{os.path.basename(self.args.model_path)} Inference"
        temp_ram_measurement(output_dir, target_name, self.args.measure_duration, self.args.measure_time_interval)

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        new_text = f"FPS: {fps:.2f}\nDroprate: {droprate:.2f}\nAvg FPS: {avgfps:.2f}"
        self.pipeline.get_by_name("text_overlay").set_property("text", new_text)
        return True

    def sink_probe(self, pad, info):
        buffer = info.get_buffer()
        if not buffer or buffer.pts == Gst.CLOCK_TIME_NONE:
            print("No valid timestamp")
            return Gst.PadProbeReturn.OK

        now = Gst.util_get_timestamp()
        latency_ns = now - buffer.pts
        latency_ms = latency_ns / 1e6

        print(f"[System Latency] {latency_ms:.2f} ms")
        return Gst.PadProbeReturn.OK

    def picamera_thread(self):
        with Picamera2() as picam2:
            if self.picamera_config is None:
                # Default configuration
                main = {'size': (self.image_width, self.image_height), 'format': 'RGB888'}
                controls = {}
                config = picam2.create_preview_configuration(main=main, controls=controls)
            else:
                config = self.picamera_config
            # Configure the camera with the created configuration
            camera_modes = picam2.sensor_modes
            
            # Print available modes for debugging
            print("Available sensor modes:")
            for i, mode in enumerate(camera_modes):
                print(f"Mode {i}: {mode}")
        
            picam2.configure(config)

            #print(f"Picamera2 configuration: width={width}, height={height}, format={format_str}")
            picam2.start()

            #Set fps so auto-exposure doesn't introduce latency
            picam2.set_controls({"FrameRate": self.args.fps})
            picam2.set_controls({"AfMode": libcamera.controls.AfModeEnum.Continuous})


            print("picamera_process started")
            while True:
                frame_data = picam2.capture_array('main')

                if frame_data is None:
                    print("Failed to capture frame.")
                    break

                frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)

                frame_pts = {
                    'frame': frame,
                    'pts': Gst.util_get_timestamp()
                }

                self.cam_queue.put((frame_pts))

    def cv2_camera_thread(self):
        # Initialize the camera
        cap = cv2.VideoCapture(0)  # 0 is usually the default camera
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

            # Check available resolutions (optional, may not work on all cameras)
        print("Attempting to check available camera modes...")
        try:
            test_resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
            for w, h in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                if abs(w - actual_w) < 10 and abs(h - actual_h) < 10:
                    print(f"Camera supports: {actual_w}x{actual_h}")
        except:
            print("Failed to test resolutions")
        
        # Print camera properties for debugging
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera initialized: width={actual_width}, height={actual_height}, fps={actual_fps}")
        
        if actual_width != self.image_width:
            self.image_width = actual_width
            self.image_height = actual_height

        print("cv2_camera_thread started")
        while True:
            # Capture frame-by-frame
            ret, frame_data = cap.read()
            
            if not ret or frame_data is None:
                print("Failed to capture frame.")
                break

            # Convert from BGR to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
   
            # Put the frame in the queue
            self.cam_queue.put((frame))
        
        # Release the camera when the loop is exited
        cap.release()

    def preprocess_thread(self, queue):
        while True:
            try:
                item = queue.get(timeout=5)

                if item is None:
                    break

                frame = item['frame']

                input_data, pad = preprocess(self, frame)
               
                self.input_queue.put(([item], [input_data]))

            except Exception as e:
                print(f"Error in preprocessing thread: {e}")
    
    def postprocess_thread(self, queue):
        while True:
            try:
                item = queue.get(timeout=5)

                original_frame_pts, infer_results = item

                original_frame = original_frame_pts['frame']

                if len(infer_results) == 1:
                    infer_results = infer_results[0]

                if self.args.model_arch == "v9":
                    postprocess_v9(self, original_frame, infer_results, True, None)
                if self.args.model_arch == "v8":
                    postprocess_v8(self, original_frame, infer_results, True, None)

                new_buffer = Gst.Buffer.new_wrapped(original_frame.tobytes())
                new_buffer.pts = int(original_frame_pts['pts'])
                
                self.appsrc.emit("push-buffer", new_buffer)
            except Exception as e:
                if type(e).__name__ == "Empty":
                    print("Waiting for inference result...")
                else:
                    error_msg = (
                        f"Error in thread: Postprocessing\n"
                        f"Type: {type(e).__name__}\n"
                        f"Message: {str(e)}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                    print(error_msg, file=sys.stderr)


    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()

        if self.args.input == "picam":
            self.cam_thread = threading.Thread(target=self.picamera_thread, args=(), daemon=True )
            self.cam_thread.start()
        else:
            self.cam_thread = threading.Thread(target=self.cv2_camera_thread, args=(), daemon=True )
            self.cam_thread.start()

        self.preprocess_thread = threading.Thread(target=self.preprocess_thread, args=(self.cam_queue,), daemon=True )
        self.preprocess_thread.start()

        self.postprocess_thread = threading.Thread(target=self.postprocess_thread, args=(self.output_queue,), daemon=True )
        self.postprocess_thread.start()

        if self.args.measure_temp_ram == True:
            self.measurement_thread = threading.Thread(target=self.measurement_temp_ram_thread, args=(), daemon=True )
            self.measurement_thread.start()
        
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

            if self.cam_thread.is_alive():
                self.cam_thread.join()
            if self.preprocess_thread.is_alive():
                self.preprocess_thread.join()
            if self.postprocess_thread.is_alive():
                self.postprocess_thread.join()

            if self.args.measure_temp_ram == True:
                if self.measurement_thread.is_alive():
                    self.measurement_thread.join()

            self.loop.quit()
            print("Cleanup complete. Exiting...")
        
if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()

