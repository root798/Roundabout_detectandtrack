import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from detector.mapper import Mapper
from tracker.ucmc import UCMCTrack


# Load annotations from JSON
def load_annotations(annotation_file):
    """Load annotations from a JSON file."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    annotations = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        x_min = min([p[0] for p in points])
        y_min = min([p[1] for p in points])
        x_max = max([p[0] for p in points])
        y_max = max([p[1] for p in points])
        annotations.append({
            "label": label,
            "bbox": [x_min, y_min, x_max, y_max]
        })
    return annotations


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def evaluate_tracking(detections, annotations, iou_threshold=0.5):
    """Evaluate tracking performance against annotations."""
    matched = 0
    for det in detections:
        for ann in annotations:
            iou = calculate_iou(det["bbox"], ann["bbox"])
            if iou > iou_threshold:
                matched += 1
                break
    precision = matched / len(detections) if detections else 0
    recall = matched / len(annotations) if annotations else 0
    return precision, recall


def get_image_files(folder):
    """Get sorted list of image files in a folder."""
    return sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])


class Detector:
    """Detector class for running YOLO detections."""
    def __init__(self, yolo_version="yolov5", cam_para_file=None):
        self.mapper = Mapper(cam_para_file, "MOT17") if cam_para_file else None
        self.model = YOLO(f'pretrained/{yolo_version}.pt')

    def get_dets(self, img, conf_thresh=0.01, det_classes=[2, 5, 7]):
        dets = []
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(frame, imgsz=2560)  # Adjust resolution as needed
        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            det = {
                "id": det_id,
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "conf": conf,
                "class": cls_id
            }
            if self.mapper:
                det["y"], det["R"] = self.mapper.mapto([bbox[0], bbox[1], w, h])
            det_id += 1
            dets.append(det)

        return dets
def visualize_detections_and_annotations(frame, detections, annotations, output_path):
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for detections
        cv2.putText(frame, f"Det: {det['class']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw annotations
    for ann in annotations:
        x1, y1, x2, y2 = map(int, ann['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for annotations
        cv2.putText(frame, f"Ann: {ann['label']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the visualization
    cv2.imwrite(output_path, frame)


def optimize_parameters(annotation_folder, image_folder, parameter_grid, yolo_version="yolov5", cam_para_file=None):
    """Optimize parameters based on ground-truth annotations."""
    best_params = None
    best_score = 0

    detector = Detector(yolo_version=yolo_version, cam_para_file=cam_para_file)

    for params in parameter_grid:
        print(f"Testing params: {params}")
        wx, wy, vmax, conf_thresh = params

        # Initialize performance metrics
        total_precision, total_recall = 0, 0
        frame_count = 0

        # Get image files
        image_files = get_image_files(image_folder)

        for frame_id, image_file in enumerate(image_files, start=1):
            # Load image
            img_path = os.path.join(image_folder, image_file)
            frame = cv2.imread(img_path)

            # Load corresponding annotation
            annotation_file = os.path.join(annotation_folder, f"{os.path.splitext(image_file)[0]}.json")
            if not os.path.exists(annotation_file):
                print(f"No annotation file for {image_file}, skipping.")
                continue
            annotations = load_annotations(annotation_file)
            #print(f"Annotations for {image_file}: {annotations}")

            # Get detections
            detections = detector.get_dets(frame, conf_thresh=conf_thresh)
            #print(f"Detections for {image_file}: {detections}")

            # Visualize detections and annotations
            output_path = f"output_vis/{image_file}"
            visualize_detections_and_annotations(frame.copy(), detections, annotations, output_path)

            # Evaluate the detections against annotations
            precision, recall = evaluate_tracking(detections, annotations, iou_threshold=0.3)
            print(f"Precision: {precision}, Recall: {recall}")

            total_precision += precision
            total_recall += recall
            frame_count += 1

        # Compute average precision and recall
        avg_precision = total_precision / frame_count if frame_count > 0 else 0
        avg_recall = total_recall / frame_count if frame_count > 0 else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-10) if avg_precision + avg_recall > 0 else 0

        print(f"Params: {params}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {f1_score:.4f}")

        # Update best parameters if current score is better
        if f1_score > best_score:
            best_score = f1_score
            best_params = params

    return best_params, best_score

# def optimize_parameters(annotation_folder, image_folder, parameter_grid, yolo_version="yolov5", cam_para_file=None):
#     """Optimize parameters based on ground-truth annotations."""
#     best_params = None
#     best_score = 0

#     detector = Detector(yolo_version=yolo_version, cam_para_file=cam_para_file)

#     for params in parameter_grid:
#         print(f"Testing params: {params}")
#         wx, wy, vmax, conf_thresh = params

#         # Initialize performance metrics
#         total_precision, total_recall = 0, 0
#         frame_count = 0

#         # Get image files
#         image_files = get_image_files(image_folder)

#         for frame_id, image_file in enumerate(image_files, start=1):
#             # Load image
#             img_path = os.path.join(image_folder, image_file)
#             frame = cv2.imread(img_path)

#             # Load corresponding annotation
#             annotation_file = os.path.join(annotation_folder, f"{os.path.splitext(image_file)[0]}.json")
#             if not os.path.exists(annotation_file):
#                 continue  # Skip if no annotation for this image
#             annotations = load_annotations(annotation_file)
#             print(f"Annotations for {image_file}: {annotations}")

#             # Get detections
#             detections = detector.get_dets(frame, conf_thresh=conf_thresh)
#             print(f"Detections for {image_file}: {detections}")

#             # Evaluate the detections against annotations
#             precision, recall = evaluate_tracking(detections, annotations)
#             print(f"Precision: {precision}, Recall: {recall}")

#             total_precision += precision
#             total_recall += recall
#             frame_count += 1

#         # Compute average precision and recall
#         avg_precision = total_precision / frame_count if frame_count > 0 else 0
#         avg_recall = total_recall / frame_count if frame_count > 0 else 0
#         f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-10) if avg_precision + avg_recall > 0 else 0

#         print(f"Params: {params}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {f1_score:.4f}")

#         # Update best parameters if current score is better
#         if f1_score > best_score:
#             best_score = f1_score
#             best_params = params

#     return best_params, best_score


# Example parameter grid for optimization
parameter_grid = [
    (10, 10, 5, 0.01),
    (20, 20, 5, 0.01),
    (15, 15, 3, 0.005),
    # Add more parameter combinations as needed
]

# Run optimization
annotation_folder = "C:/research/180-250/180-250"  # Replace with the actual path
image_folder = "C:/research/180-250/180-250"  # Replace with the actual path
best_params, best_score = optimize_parameters(annotation_folder, image_folder, parameter_grid, yolo_version="yolov8x")

print(f"Best Parameters: {best_params}, Best Score: {best_score}")
