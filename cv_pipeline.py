from gi.repository import Gst, GLib
from picamera2 import Picamera2
import cv2
import threading

class GstOpenCVPipeline:
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)
        self.picamera_config = None
        self.image_width = 1920
        self.image_height = 1080

        # Define the sink pipeline
        sink_pipeline_str = (
            "appsrc name=opencv_src format=time is-live=true do-timestamp=true block=false "
            f'caps=video/x-raw,format=RGB,width={self.image_width},height={self.image_height} ! '
            "queue max-size-buffers=3 leaky=downstream ! "
            'autovideoconvert !  textoverlay name=text_overlay text=" " valignment=bottom halignment=center font-desc="Arial, 36" !'
            "fpsdisplaysink name=display video-sink=autovideosink sync=true text-overlay=true signal-fps-measurements=true"
        )

        self.pipeline = Gst.parse_launch(sink_pipeline_str)
        #self.appsrc = self.sink_pipeline.get_by_name("opencv_src")

    def picamera_thread(self):
        appsrc = self.pipeline.get_by_name("opencv_src")
        appsrc.set_property("is-live", True)
        appsrc.set_property("format", Gst.Format.TIME)
        print("appsrc properties: ", appsrc)

        with Picamera2() as picam2:
            if self.picamera_config is None:
                # Default configuration
                main = {'size': (self.video_width, self.video_height), 'format': 'RGB888'}
                lores = {'size': (self.video_width, self.video_height), 'format': 'RGB888'}
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
            appsrc.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw, format={format_str}, width={width}, height={height}, "
                    f"pixel-aspect-ratio=1/1"
                )
            )
            picam2.start()

            print("picamera_process started")
            while True:
                frame_data = picam2.capture_array('lores')

                if frame_data is None:
                    print("Failed to capture frame.")
                    break
                # Convert framontigue data if necessary
                # frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                # frame = np.asarray(frame)
                # Create Gst.Buffer by wrapping the frame data
                buffer = Gst.Buffer.new_wrapped(frame_data.tobytes())

                # Push the buffer to appsrc
                ret = appsrc.emit('push-buffer', buffer)
                if ret != Gst.FlowReturn.OK:
                    print("Failed to push buffer:", ret)
                    break

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()

        self.picam_thread = threading.Thread(target=self.picamera_thread, args=(), daemon=True )
        self.picam_thread.start()

        try:
            # Run the main loop
            self.loop.run()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Shutting down...")
        finally:
               
            # Clean up
            self.pipeline.set_state(Gst.State.NULL)

            if self.picam_thread.is_alive():
                self.picam_thread.join()

            self.loop.quit()
            print("Cleanup complete. Exiting...")
        
if __name__ == "__main__":
    pipeline = GstOpenCVPipeline()
    pipeline.run()

