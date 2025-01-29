import os
import glob
import json
import math
import argparse
import numpy as np
import motmetrics as mm

###############################################################################
# Part 1: Simpler Detection Metrics (IoU-based)
###############################################################################
def iou_rect(rect1, rect2):
    """
    Computes IoU between two rectangles.
    rect = [x1, y1, x2, y2]
    """
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def get_rect_coords(points):
    """
    Takes a shape's points in [ [x1, y1], [x2, y1], [x2, y2], [x1, y2] ]
    and returns [xmin, ymin, xmax, ymax].
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [xmin, ymin, xmax, ymax]

def load_json_bboxes(json_path):
    """
    Loads bounding boxes and track_ids from a given JSON.
    Returns a list of (rect_coords, track_id).
    rect_coords = [xmin, ymin, xmax, ymax]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    bboxes = []
    shapes = data.get("shapes", [])
    for shape in shapes:
        points = shape.get("points", [])
        rect = get_rect_coords(points)
        track_id = shape.get("group_id", -1)  # default to -1 if not present
        bboxes.append((rect, track_id))
    return bboxes

def compute_detection_metrics(gt_folder, pred_folder, iou_threshold=0.5):
    """
    Compares all JSON files in gt_folder and pred_folder.
    Computes:
    - TP, FP, FN
    - Precision, Recall, F1
    - Simple bounding box area MAPE for matched boxes
    - Basic tracking ID consistency (raw ID matches)
    """
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.json")))
    if not gt_files:
        print("No ground-truth JSON files found.")
        return

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # For bounding box area MAPE
    area_percentage_errors = []

    # For basic tracking metric
    total_matched_tracks = 0
    total_track_comparisons = 0

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(pred_folder, filename)
        if not os.path.exists(pred_path):
            # If a predicted file does not exist for this ground-truth frame, count all GT as missed
            gt_bboxes = load_json_bboxes(gt_path)
            total_fn += len(gt_bboxes)
            continue

        gt_bboxes = load_json_bboxes(gt_path)
        pred_bboxes = load_json_bboxes(pred_path)

        # Make a list of unmatched ground-truth indices (for calculating FN later)
        unmatched_gt_indices = set(range(len(gt_bboxes)))

        # Tally for the predicted bounding boxes
        frame_tp = 0
        frame_fp = 0

        # We will match each predicted box to the GT box with highest IoU > iou_threshold
        used_gt = set()
        for (pred_rect, pred_track_id) in pred_bboxes:
            best_iou = 0.0
            best_gt_idx = -1

            for idx, (gt_rect, gt_track_id) in enumerate(gt_bboxes):
                if idx in used_gt:
                    continue
                iou_val = iou_rect(pred_rect, gt_rect)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = idx

            # If IoU is high enough, this is a true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                frame_tp += 1
                used_gt.add(best_gt_idx)
                unmatched_gt_indices.discard(best_gt_idx)

                # Compute area-based MAPE for matched pair
                gt_rect, gt_tid = gt_bboxes[best_gt_idx]
                gt_area = (gt_rect[2] - gt_rect[0]) * (gt_rect[3] - gt_rect[1])
                pred_area = (pred_rect[2] - pred_rect[0]) * (pred_rect[3] - pred_rect[1])
                if gt_area > 0.0:
                    area_err = abs(pred_area - gt_area) / gt_area
                    area_percentage_errors.append(area_err)

                # Track ID matching check
                total_track_comparisons += 1
                if pred_track_id == gt_tid:
                    total_matched_tracks += 1

            else:
                # IoU too low -> false positive
                frame_fp += 1

        # False negatives are any GT boxes not matched
        frame_fn = len(unmatched_gt_indices)

        total_tp += frame_tp
        total_fp += frame_fp
        total_fn += frame_fn

    # Compute detection metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute MAPE for bounding box areas
    if area_percentage_errors:
        mean_area_mape = sum(area_percentage_errors) / len(area_percentage_errors) * 100.0
    else:
        mean_area_mape = 0.0

    # Compute raw tracking ID match rate
    if total_track_comparisons > 0:
        track_id_accuracy = total_matched_tracks / total_track_comparisons
    else:
        track_id_accuracy = 0.0

    # Print detection metrics
    print("Detection Metrics (IoU-Based):")
    print(f"  TP: {total_tp}")
    print(f"  FP: {total_fp}")
    print(f"  FN: {total_fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1_score:.3f}")
    print()
    print("Area MAPE (for matched boxes):")
    print(f"  {mean_area_mape:.2f}%")
    print()
    print("Tracking ID Consistency (Raw ID Match):")
    print(f"  Matched Tracks: {total_matched_tracks} / {total_track_comparisons}")
    print(f"  Track ID Accuracy: {track_id_accuracy:.3f}")
    print("-----------------------------------------------------------")

###############################################################################
# Part 2: Advanced Tracking Metrics (motmetrics)
###############################################################################
def load_mot_format(json_path):
    """
    Return rows of [x_left, y_top, width, height, track_id].
    """
    data = []
    with open(json_path, 'r') as f:
        j = json.load(f)
    shapes = j.get("shapes", [])
    for shape in shapes:
        pts = shape["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        track_id = shape.get("group_id", -1)
        width = x2 - x1
        height = y2 - y1
        data.append([x1, y1, width, height, track_id])
    return data

def compute_mot_metrics(gt_folder, pred_folder):
    """
    Uses motmetrics to compute advanced multi-object tracking metrics such as:
    MOTA, IDF1, IDP, IDR, and number of ID switches.
    """
    acc = mm.MOTAccumulator(auto_id=True)

    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.json")))
    pred_files = sorted(glob.glob(os.path.join(pred_folder, "*.json")))

    # For each pair of GT and prediction files
    for frame_idx, (gt_file, pred_file) in enumerate(zip(gt_files, pred_files), start=1):
        gt_data = load_mot_format(gt_file)
        pred_data = load_mot_format(pred_file)

        # Build arrays for ground truth
        gt_positions = []
        gt_ids = []
        for (x, y, w, h, tid) in gt_data:
            # Use bounding box center for distance
            gt_positions.append([x + w/2, y + h/2])
            gt_ids.append(tid)

        # Build arrays for predictions
        pred_positions = []
        pred_ids = []
        for (x, y, w, h, tid) in pred_data:
            pred_positions.append([x + w/2, y + h/2])
            pred_ids.append(tid)

        # If both GT and predicted are empty, skip
        if len(gt_positions) == 0 and len(pred_positions) == 0:
            acc.update([], [], [])
            continue

        gt_positions = np.array(gt_positions)
        pred_positions = np.array(pred_positions)

        # Squared Euclidean distance, set distances over (100^2) to NaN
        cost_matrix = mm.distances.norm2squared_matrix(gt_positions, pred_positions, max_d2=100**2)

        acc.update(
            gt_ids,      # ground-truth IDs
            pred_ids,    # predicted IDs
            cost_matrix  # NxM cost matrix
        )

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['num_frames', 'mota', 'idf1', 'idp', 'idr', 'num_switches']
    )

    print("Advanced Tracking Metrics (motmetrics):")
    print(summary)

###############################################################################
# Main: Run Both Sets of Metrics
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Compare JSON files for detection and tracking metrics.")
    parser.add_argument("--gt_folder", type=str, default="C:/research/Archive/180-250/180-250", help="Path to folder with ground-truth JSON files")
    parser.add_argument("--pred_folder", type=str, default="C:/research/UCMCTrack/output_json", help="Path to folder with predicted JSON files")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching bounding boxes")

    args = parser.parse_args()

    # 1) Compute the simpler IoU-based detection metrics
    compute_detection_metrics(args.gt_folder, args.pred_folder, iou_threshold=args.iou_threshold)

    # 2) Compute advanced tracking metrics using motmetrics
    compute_mot_metrics(args.gt_folder, args.pred_folder)

if __name__ == "__main__":
    main()
