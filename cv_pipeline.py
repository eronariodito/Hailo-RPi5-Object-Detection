import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from picamera2 import Picamera2
import cv2
import threading
import numpy as np
import queue

class GstOpenCVPipeline:
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)
        self.picamera_config = None
        self.image_width = 1920
        self.image_height = 1080

        self.cam_queue = queue.Queue(maxsize=1)

        # Define the sink pipeline
        sink_pipeline_str = (
            "appsrc name=opencv_src format=time is-live=true do-timestamp=true block=false "
            f'caps=video/x-raw,format=RGB,width={self.image_width},height={self.image_height} ! '
            "queue max-size-buffers=3 leaky=downstream ! "
            'videoconvert !  textoverlay name=text_overlay text=" " valignment=bottom halignment=center font-desc="Arial, 36" !'
            "fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=true signal-fps-measurements=true"
        )

        sink_pipeline_str = (
            "appsrc name=opencv_src format=time is-live=true do-timestamp=true block=false "
            f'caps=video/x-raw,format=RGB,width=640,height=640 ! '
            "queue max-size-buffers=3 leaky=downstream ! "
            'videoconvert !  textoverlay name=text_overlay text=" " valignment=bottom halignment=center font-desc="Arial, 36" !'
            "fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=true signal-fps-measurements=true"
        )

        self.pipeline = Gst.parse_launch(sink_pipeline_str)
        self.appsrc = self.pipeline.get_by_name("opencv_src")

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
            # Update GStreamer caps based on 'lores' stream
            lores_stream = config['lores']
            format_str = 'RGB' if lores_stream['format'] == 'RGB888' else self.video_format
            width, height = lores_stream['size']
            print(f"Picamera2 configuration: width={width}, height={height}, format={format_str}")
            picam2.start()

            print("picamera_process started")
            while True:
                frame_data = picam2.capture_array('lores')

                if frame_data is None:
                    print("Failed to capture frame.")
                    break
                # Convert framontigue data if necessary
                frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                # Create Gst.Buffer by wrapping the frame data
                # buffer = Gst.Buffer.new_wrapped(frame_data.tobytes())

                # # Push the buffer to appsrc
                # ret = self.appsrc.emit('push-buffer', buffer)
                # if ret != Gst.FlowReturn.OK:
                #     print("Failed to push buffer:", ret)
                #     break
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

                # Create a new Gst.Buffer from the flipped frame without copying the memory
                new_buffer = Gst.Buffer.new_wrapped(input_data.tobytes())

                self.appsrc.emit("push-buffer", new_buffer)
            except Exception as e:
                print(f"Error in preprocessing thread: {e}")
                
    def preprocess(self, frame):
        
        # frame_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        frame, pad = self.letterbox(frame, (640, 640))
        frame = frame[None]
        frame = np.ascontiguousarray(frame)
        frame = frame.astype(np.float32)
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

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()

        self.picam_thread = threading.Thread(target=self.picamera_thread, args=(), daemon=True )
        self.picam_thread.start()

        self.preprocess_thread = threading.Thread(target=self.preprocess_thread, args=(self.cam_queue,), daemon=True )
        self.preprocess_thread.start()

        try:
            # Run the main loop
            self.loop.run()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Shutting down...")
        finally:
               
            # Clean up
            self.pipeline.set_state(Gst.State.NULL)

            self.cam_queue.put(None)

            if self.picam_thread.is_alive():
                self.picam_thread.join()
            if self.preprocess_thread.is_alive():
                self.preprocess_thread.join()

            self.loop.quit()
            print("Cleanup complete. Exiting...")
        
if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()

