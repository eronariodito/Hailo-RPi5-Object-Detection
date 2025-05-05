import argparse

def setup_parser_benchmark():
    parser = argparse.ArgumentParser(description="Configure model parameters")

    # Add arguments for each configurable variable
    parser.add_argument('--conf-all', type=float, default=0.001, help="Confidence threshold for all detections (default: 0.001)")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold for plotting detection (default: 0.25)")
    parser.add_argument('--iou', type=float, default=0.7, help="Intersection over union (IoU) threshold for NMS (default: 0.7)")
    parser.add_argument('--model-path', type=str, default="models/v9/v9-s_640.hef", help="Path to model hef")  
    parser.add_argument('--model-arch', type=str,choices=['v8', 'v9'], help="YOLO model architecture")
    parser.add_argument('--dataset', type=str, default="datasets/coco/coco.yaml", help="Dataset YAML file (default: 'datasets/coco/coco.yaml')")

    return parser

def parse_arguments_benchmark():
    parser = setup_parser_benchmark()
    return parser.parse_args()

def setup_parser_inference():
    parser = argparse.ArgumentParser(description="Configure model parameters")

    # Add arguments for each configurable variable
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument('--iou', type=float, default=0.45, help="Intersection over union (IoU) threshold for NMS (default: 0.45)")
    parser.add_argument('--model-path', type=str, default="models/v9/v9-s_640.hef", help="Path to model hef")  
    parser.add_argument('--model-arch', type=str,choices=['v8', 'v9'], help="YOLO model architecture")
    parser.add_argument('--dataset', type=str, default="datasets/coco/coco.yaml", help="Dataset YAML file for label(default: 'datasets/coco/coco.yaml')")
    parser.add_argument('--camera-res', type=str, default="1920,1080", help="Setup the camera resolution (default: '1920,1080')")
    parser.add_argument('--fps', type=int, default="30", help="Setup the camera fps (default: '30')")
    parser.add_argument('--input', type=str, default="picam", choices=['picam', 'opencv'], help="Select input camera (default: 'picam')")
    parser.add_argument('-mtr', '--measure-temp-ram', action='store_true', help="Save temperature and ram measurement (default: 'False')")
    parser.add_argument('-md', '--measure-duration', type=int, default="600", help="Set measurement duration for temperature and ram in seconds (default: '600')")
    parser.add_argument('-mti', '--measure-time-interval', type=int, default="10", help="Set measurement interval for temperature and ram in seconds (default: '10')")


    return parser

def parse_arguments_inference():
    parser = setup_parser_inference()
    return parser.parse_args()

def setup_parser_video():
    parser = argparse.ArgumentParser(description="Configure model parameters")

    # Add arguments for each configurable variable
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument('--iou', type=float, default=0.45, help="Intersection over union (IoU) threshold for NMS (default: 0.45)")
    parser.add_argument('--int8', type=bool, default=True, help="Use INT8 quantization (default: True)")
    parser.add_argument('--model-path', type=str, default="test_checkpoint_full_integer_quant.tflite", help="Path to model hef")  
    parser.add_argument('--model-arch', type=str,choices=['v8', 'v9'], help="YOLO model architecture")
    parser.add_argument('--dataset', type=str, default="datasets/coco/coco.yaml", help="Dataset YAML file for label(default: 'coco.yaml')")
    parser.add_argument('--video-path', type=str, default="videoplayback.mp4", help="Path to video file (default: 'videoplayback.mp4')")  

    return parser

def parse_arguments_video():
    parser = setup_parser_video()
    return parser.parse_args()