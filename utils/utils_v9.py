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

def postprocess_v9(self, original_frame, infer_results, plot, image_id):
    image_size = (self.input_width, self.input_height)

    rev_tensor = get_rev_tensor(original_frame, image_size)

    model_outputs, layer_output = [], []

    idx = 0
    for key, predict in infer_results.items():
        if idx % 3 == 1:
            predict = predict.reshape(1, predict.shape[1], predict.shape[2], 16,4)
        layer_output.append(predict)
        if idx % 3 == 2:
            model_outputs.append(layer_output)
            layer_output = []
        idx += 1
    if len(model_outputs) == 6:
        model_outputs = model_outputs[:3]

    # for item in model_outputs:
    #     for item2 in item:
    #         print(item2.shape)

    preds_cls, preds_anc, preds_box = [], [], []

    for layer_output in model_outputs:
        pred_cls, pred_anc, pred_box = layer_output
        B, w, h, C = pred_cls.shape
        _, _, _, A, R = pred_anc.shape
        _, _, _, X = pred_box.shape

        # Reshape like einops
        preds_cls.append(pred_cls.transpose(0, 1, 2, 3).reshape(B, h * w, C))         # [B, h*w, C]
        preds_anc.append(pred_anc.transpose(0, 1, 2, 3, 4).reshape(B, h * w, R, A))    # [B, h*w, R, A]
        preds_box.append(pred_box.transpose(0, 1, 2, 3).reshape(B, h * w, X))          # [B, h*w, X]

    # Concatenate predictions from all layers
    preds_cls = np.concatenate(preds_cls, axis=1)  # [B, total_anchors, C]
    preds_anc = np.concatenate(preds_anc, axis=1)  # [B, total_anchors, R, A]
    preds_box = np.concatenate(preds_box, axis=1)  # [B, total_anchors, X]

    strides = [8, 16, 32]
    all_anchors, all_scalers = generate_anchor_v9_numpy(image_size, strides)

    all_scalers_reshaped = all_scalers.reshape(1, -1, 1)
    pred_LTRB = preds_box * all_scalers_reshaped

    lt = pred_LTRB[..., :pred_LTRB.shape[-1]//2]  # First half of the last dimension
    rb = pred_LTRB[..., pred_LTRB.shape[-1]//2:]  # Second half of the last dimension

    # Equivalent to torch.cat - in NumPy we use np.concatenate
    preds_box = np.concatenate([all_anchors - lt, all_anchors + rb], axis=-1)

    # Create the prediction tuple
    prediction = (preds_cls, preds_anc, preds_box)

    # Extract from the tuple
    pred_class, _, pred_bbox = prediction[:3]

    if len(prediction) == 4:
        pred_conf = prediction[3]
    else:
        pred_conf = None

    if rev_tensor is not None:
        pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]

    indices, boxes, scores, class_ids = box_nms_v9(self, original_frame, pred_class, pred_bbox, plot)

    if image_id != None:
        result =  postprocess_json(self, indices, boxes, scores, class_ids, image_id)
    else:
        result = 0

    return result

            
def box_nms_v9(self, frame, cls_dist, bbox, plot) :
    """
    Process model outputs to extract and visualize detections.

    Args:
        img (np.ndarray): The original input image.
        outputs (np.ndarray): Raw model outputs.
        pad (Tuple[float, float]): Padding ratios from preprocessing.

    Returns:
        (np.ndarray): The input image with detections drawn on it.
    """
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    cls_dist = sigmoid(cls_dist.astype(np.float32)) #* (1 if confidence is None else confidence)

    batch_idx, valid_grid, valid_cls = np.where(cls_dist > self.conf_all)
    valid_con = cls_dist[batch_idx, valid_grid, valid_cls]
    valid_box = bbox[batch_idx, valid_grid]

    # Split into components
    xmin = valid_box[:, 0]
    ymin = valid_box[:, 1]
    xmax = valid_box[:, 2]
    ymax = valid_box[:, 3]

    # Ensure x1 < x2 and y1 < y2
    x1 = np.minimum(xmin, xmax)
    x2 = np.maximum(xmin, xmax)
    y1 = np.minimum(ymin, ymax)
    y2 = np.maximum(ymin, ymax)

    # Calculate width and height
    w = x2 - x1
    h = y2 - y1

    valid_box = np.stack([x1, y1, w, h], axis=1)

    indices = cv2.dnn.NMSBoxes(valid_box, valid_con, self.conf_all, self.iou)

    if plot:
        if isinstance(indices, (list, np.ndarray)) and len(indices) > 0:
                indices.flatten()
        else:
            return 0, 0, 0, 0
        # Draw detections that survived NMS
        [draw_detections_v9(self,frame, valid_box[i], valid_con[i], valid_cls[i]) for i in indices]

    return indices, valid_box, valid_con, valid_cls

def get_rev_tensor(frame, new_shape):
    shape = frame.shape[:2]  # Current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_width, new_height = int(shape[1] * r), int(shape[0] * r)

    pad_left = (new_shape[0] - new_width) // 2
    pad_top = (new_shape[1] - new_height) // 2

    return np.array([[r, pad_left, pad_top, pad_left, pad_top]], dtype=np.float32)

def generate_anchor_v9_numpy(image_size, strides):
    W, H = image_size
    anchors = []
    scaler = []

    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(np.full((anchor_num,), stride))
        shift = stride // 2
        h = np.arange(0, H, stride) + shift
        w = np.arange(0, W, stride) + shift

        anchor_h, anchor_w = np.meshgrid(h,w, indexing='ij')
        anchor = np.stack([anchor_w.flatten(), anchor_h.flatten()], axis=-1)

        anchors.append(anchor)

    all_anchors = np.concatenate(anchors, axis=0)
    all_scalers = np.concatenate(scaler, axis=0)

    return all_anchors, all_scalers

def draw_detections_v9(self, frame, box, score, class_id):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """
    if score < self.conf:
        return
    # Extract the coordinates of the bounding box
    
    x1,y1,w,h = box
    

    # Retrieve the color for the class ID
    color = self.color_palette[class_id]

    # Draw the bounding box on the image
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # Create the label text with class name and score
    label = f"{self.classes[class_id]}: {score:.2f}"

    font_scale = min(frame.shape[0], frame.shape[1]) / 1000

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        frame,
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

    # Draw the label text on the image
    cv2.putText(frame, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)

def postprocess_json(self, indices, boxes, scores, class_ids, image_id) :
    """
    Process model outputs to extract and visualize detections.

    Args:
        img (np.ndarray): The original input image.
        outputs (np.ndarray): Raw model outputs.
        pad (Tuple[float, float]): Padding ratios from preprocessing.

    Returns:
        (np.ndarray): The input image with detections drawn on it.
    """
    coco_json = []

    if isinstance(indices, (list, np.ndarray)) and len(indices) > 0:
        indices.flatten()
    else:
        return 

    # Draw detections that survived NMS
    for i in indices:
        x1,y1,w,h = boxes[i]

        coco_json.append({
                "image_id": int(image_id),
                "category_id": self.coco_mapping_80to91[f"{class_ids[i] + 1}"],
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(scores[i])
            })

    return coco_json
