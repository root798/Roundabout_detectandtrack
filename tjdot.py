import numpy as np
import cv2
import csv
import os
import glob
from ultralytics import YOLO
from tracker.ucmc import UCMCTrack
from tracker.kalman import KalmanTracker
from detector.mapper import Mapper
from collections import defaultdict
import argparse

# Detection class
class Detection:
    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(2)

# Detector class
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self, cam_para_file, yolo_version):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.model = YOLO(f'pretrained/{yolo_version}.pt')

    def get_dets(self, img, conf_thresh=0, det_classes=[0]):
        dets = []
        # Convert BGR to RGB for YOLO
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Run the detection
        results = self.model(frame, imgsz=2560)
        det_id = 0
        for box in results[0].boxes:
            conf = float(box.conf.cpu().numpy()[0])
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            # Filter small boxes, class mismatch, and low conf
            if (w <= 10 and h <= 10) or (cls_id not in det_classes) or (conf <= conf_thresh):
                continue

            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            # Map to real-world coordinates if desired
            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1
            dets.append(det)

        return dets

def main(args):
    # Initialize detector and tracker
    yolo_version = "yolo11x"
    detector = Detector()
    detector.load(args.cam_para, yolo_version)
    tracker = UCMCTrack(
        args.a,
        args.a,
        args.wx,
        args.wy,
        args.vmax,
        args.cdt,
        args.fps,
        "MOT",
        args.high_score,
        False,
        None
    )

    # Output folder for CSV files
    os.makedirs("trajectories_csv", exist_ok=True)

    # Gather list of images (e.g., 00001.jpg, 00002.jpg, etc.)
    image_files = sorted(glob.glob(os.path.join(args.images_folder, "*.jpg")))

    frame_id = 1
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}. Skipping.")
            continue

        # Detection
        dets = detector.get_dets(img, args.conf_thresh, [2, 5, 7])  # e.g. 2=Car,5=Bus,7=Truck
        # Tracking
        tracker.update(dets, frame_id)

        # Prepare data for CSV
        frame_data = []
        for det in dets:
            # Append row: [car_id, x_coord, y_coord, car_type_id]
            track_id = det.track_id
            x_coord = det.y[0, 0]
            y_coord = det.y[1, 0]
            class_id = det.det_class
            frame_data.append([track_id, x_coord, y_coord, class_id])

        # Write CSV for this frame
        csv_filename = f"frame_{frame_id:05d}.csv"
        csv_path = os.path.join("trajectories_csv", csv_filename)
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["car_id", "x_coordinate", "y_coordinate", "car_type_id"])
            for row in frame_data:
                writer.writerow(row)

        frame_id += 1

def parse_args():
    parser = argparse.ArgumentParser(description="Process sequential images and output tracking CSV files.")
    parser.add_argument(
        "--images_folder",
        type=str,
        default="C:/research/Archive/DePere",
        help="Path to folder containing sequential images like 00001.jpg, 00002.jpg, etc."
    )
    parser.add_argument("--cam_para", type=str, default="C:/research/UCMCTrack/demo/cam_para.txt", help="Path to camera parameter file")
    parser.add_argument("--wx", type=float, default=10, help="wx")
    parser.add_argument("--wy", type=float, default=10, help="wy")
    parser.add_argument("--vmax", type=float, default=15, help="vmax")
    parser.add_argument("--a", type=float, default=30.0, help="assignment threshold")
    parser.add_argument("--cdt", type=float, default=0.1, help="coasted deletion time")
    parser.add_argument("--high_score", type=float, default=0.3, help="high score threshold")
    parser.add_argument("--conf_thresh", type=float, default=0.3, help="detection confidence threshold")
    parser.add_argument("--fps", type=float, default=30, help="frames per second")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
