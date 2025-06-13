import numpy as np
import cv2

def preprocess(self, frame):
    frame, pad = letterbox(frame, (self.input_height, self.input_width))
    frame = frame.astype(np.uint8)
    frame = np.ascontiguousarray(frame)

    return frame, pad

def letterbox(
    frame, new_shape):
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

def postprocess_v8(self, original_frame, infer_results, plot, image_id):
    result = extract_detections_json(self, infer_results, image=original_frame, image_id=image_id)
    detections = extract_detections(infer_results, threshold = self.conf_all)

    if plot:
        draw_detections(self, detections, original_frame, min_score = self.conf)

    return result


def extract_detections(input_data: list, threshold: float = 0.5) -> dict:
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
    

    for idx in range(detections['num_detections']):
        if scores[idx] >= min_score:
            np.random.seed(classes[idx])
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            #scaled_box = boxes
            draw_detection(self, image, boxes[idx], classes[idx], scores[idx] * 100.0, color, scale_factor)

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
    label = f"{self.classes[cls]}: {score/100:.2f}"
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


def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
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

def extract_detections_json(self, input_data: list, image: np.ndarray, image_id: int = 0) -> dict:
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

            if score >= self.conf_all:
                scaled_box = denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)

                if image_id != None:
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