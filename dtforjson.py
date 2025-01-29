import numpy as np
import cv2
import glob
import os
import json
import argparse
from ultralytics import YOLO
from tracker.ucmc import UCMCTrack
from tracker.kalman import KalmanTracker
from detector.mapper import Mapper

# Define the ROI points as a numpy array if you still need it.
# If you no longer need the ROI check, you can remove it.
roi_points = np.array([[ 977,  298],
        [ 956,  297],
        [ 928,  289],
        [ 904,  289],
        [ 870,  298],
        [ 846,  306],
        [ 792,  320],
        [ 745,  326],
        [ 678,  329],
        [ 633,  329],
        [ 600,  329],
        [ 588,  342],
        [ 556,  360],
        [ 533,  372],
        [ 495,  387],
        [ 461,  397],
        [ 475,  413],
        [ 495,  433],
        [ 519,  450],
        [ 542,  468],
        [ 554,  491],
        [ 552,  532],
        [ 539,  581],
        [ 515,  605],
        [ 525,  619],
        [ 585,  623],
        [ 627,  628],
        [ 689,  632],
        [ 730,  640],
        [ 777,  661],
        [ 815,  674],
        [ 844,  685],
        [ 878,  649],
        [ 929,  630],
        [ 973,  611],
        [1015,  593],
        [1069,  583],
        [1122,  586],
        [1168,  582],
        [1234,  586],
        [1291,  590],
        [1370,  601],
        [1420,  605],
        [1447,  607],
        [1450,  580],
        [1454,  549],
        [1444,  529],
        [1419,  519],
        [1358,  512],
        [1341,  504],
        [1344,  484],
        [1345,  468],
        [1347,  452],
        [1344,  443],
        [1305,  434],
        [1276,  426],
        [1242,  410],
        [1222,  397],
        [1199,  376],
        [1163,  368],
        [1139,  355],
        [1130,  348],
        [1118,  330],
        [1113,  325],
        [1107,  309],
        [1089,  297],
        [1053,  287],
        [1024,  284],
        [1005,  284],
        [ 989,  294]], np.int32)

# Detection class (unchanged)
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

# Detector class (unchanged, except for any path changes you need)
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self, cam_para_file, yolo_version):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.model = YOLO(f'pretrained/{yolo_version}.pt')

    def get_dets(self, img, conf_thresh=0, det_classes=[0]):
        dets = []
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(frame, imgsz=2560)  # Adjust if needed
        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if (w <= 10 and h <= 10) or (cls_id not in det_classes) or (conf <= conf_thresh):
                continue

            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1
            dets.append(det)

        return dets

# Example function if you still want to check points inside ROI.
# If not needed, you can remove calls to this function.
def point_in_roi(point, roi_pts):
    # Returns True if point is inside the polygon
    return cv2.pointPolygonTest(roi_pts, point, False) >= 0

def main(args):
    # Example YOLO weights name (change as needed)
    yolo_version = "yolo11x"

    # Initialize the detector and tracker
    detector = Detector()
    detector.load(args.cam_para, yolo_version)

    tracker = UCMCTrack(
        args.a, args.a, args.wx, args.wy, args.vmax,
        args.cdt, args.fps, "MOT", args.high_score, False, None
    )

    # Gather all images named 00001.jpg, 00002.jpg, ... in the folder
    images = sorted(glob.glob(os.path.join(args.images_folder, "*.jpg")))

    frame_id = 1
    for image_path in images:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read {image_path}")
            continue

        # Get detections
        dets = detector.get_dets(img, args.conf_thresh, [2, 5, 7])  # Cars, buses, trucks
        # Update the tracker
        tracker.update(dets, frame_id)

        # Build shapes for JSON
        shapes = []
        for det in dets:
            # If you only want bounding boxes inside ROI, you can filter them here:
            center_x = int(det.bb_left + det.bb_width / 2)
            center_y = int(det.bb_top + det.bb_height / 2)
            inside_roi = point_in_roi((center_x, center_y), roi_points)

            # If you must skip bounding boxes outside ROI, uncomment below:
            # if not inside_roi:
            #     continue

            # Build the rectangle corners
            x1 = float(det.bb_left)
            y1 = float(det.bb_top)
            x2 = float(det.bb_left + det.bb_width)
            y2 = float(det.bb_top + det.bb_height)

            # Construct one shape dictionary
            shape_dict = {
                "kie_linking": [],
                "label": "car",            # If you always want "car", keep this.
                "score": None,             # Use None for null in Python
                "points": [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ],
                "group_id": int(det.track_id),  # or det.id if you prefer
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {}
            }
            shapes.append(shape_dict)

        # Prepare the final JSON structure
        json_data = {
            "version": "2.4.4",
            "flags": {},
            "shapes": shapes
        }

        # Create a matching output file name
        # For example, if image_path = ".../00001.jpg", output ".../00001.json"
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_json_path = os.path.join(args.output_folder, f"{base_name}.json")

        # Write the JSON file
        with open(output_json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        frame_id += 1

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a sequence of images and output JSON files.')
    # Folder that contains images named 00001.jpg, 00002.jpg, etc.
    parser.add_argument('--images_folder', type=str, default="C:/research/Archive/180-250/180-250", help='Folder containing input images')
    parser.add_argument('--output_folder', type=str, default="C:/research/UCMCTrack/output_json", help='Folder for JSON output')
    parser.add_argument('--cam_para', type=str, default="C:/research/UCMCTrack/demo/cam_para.txt", help='Camera parameter file')
    parser.add_argument('--wx', type=float, default=10, help='wx')
    parser.add_argument('--wy', type=float, default=10, help='wy')
    parser.add_argument('--vmax', type=float, default=15, help='vmax')
    parser.add_argument('--a', type=float, default=30.0, help='Assignment threshold')
    parser.add_argument('--cdt', type=float, default=0.1, help='Coasted deletion time')
    parser.add_argument('--high_score', type=float, default=0.3, help='High score threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--fps', type=float, default=30, help='Frames per second')
    args = parser.parse_args()

    # Create output folder if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

    main(args)
