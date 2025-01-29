import numpy as np
import cv2
import json
import os
from ultralytics import YOLO
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import argparse
# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self,cam_para_file, yolo_version):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.model = YOLO(f'pretrained/{yolo_version}.pt')#('pretrained/yolov8x.pt')

    def get_dets(self, img,conf_thresh = 0,det_classes = [0]):
        
        dets = []

        # 将帧从 BGR 转换为 RGB（因为 OpenCV 使用 BGR 格式）  
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # 使用 RTDETR 进行推理  
        results = self.model(frame,imgsz = 1088)

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id  = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # 新建一个Detection对象
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
            det_id += 1

            dets.append(det)

        return dets
    
# Define the ROI points as a numpy array for new 30s
# roi_points = np.array([
#     [28, 1046], [109, 977], [269, 846], [426, 715], [507, 614],
#     [570, 520], [521, 456], [427, 411], [308, 392], [182, 372],
#     [39, 350], [196, 251], [338, 273], [518, 299], [663, 315],
#     [805, 319], [920, 292], [1045, 236], [1099, 195], [1155, 154],
#     [1243, 103], [1311, 114], [1256, 176], [1205, 249], [1198, 346],
#     [1286, 433], [1454, 500], [1634, 555], [1786, 595], [1902, 621],
#     [1888, 797], [1665, 738], [1408, 684], [1184, 650], [1012, 680],
#     [856, 774], [739, 921], [659, 1047]
# ], np.int32)
roi_points = np.array([
    [1065, 174], [1047, 187], [1017, 206], [988, 231], [969, 243],
    [929, 284], [895, 307], [816, 327], [743, 334], [647, 341],
    [588, 332], [521, 328], [466, 318], [380, 311], [298, 303],
    [239, 297], [186, 287], [137, 287], [73, 335], [124, 341],
    [160, 344], [208, 352], [259, 359], [295, 359], [348, 368],
    [372, 372], [425, 378], [469, 386], [507, 400], [543, 420],
    [573, 446], [611, 485], [604, 528], [573, 597], [539, 641],
    [497, 697], [449, 744], [397, 790], [363, 827], [269, 922],
    [184, 1004], [108, 1075], [235, 1078], [414, 1079], [550, 1078],
    [579, 1025], [608, 967], [632, 923], [657, 874], [690, 833],
    [722, 786], [759, 731], [786, 687], [850, 644], [895, 619],
    [953, 590], [1014, 571], [1089, 556], [1190, 569], [1265, 573],
    [1332, 579], [1392, 590], [1447, 596], [1540, 612], [1631, 632],
    [1679, 645], [1775, 658], [1841, 691], [1890, 702], [1895, 584],
    [1832, 569], [1574, 521], [1448, 493], [1298, 455], [1202, 414],
    [1151, 389], [1099, 356], [1083, 309], [1093, 258], [1107, 227],
    [1122, 192], [1137, 172], [1146, 152], [1160, 136], [1141, 129],
    [1117, 127], [1097, 129], [1070, 125], [1070, 143]
], np.int32)
def log_detection(detections, frame_id, log_file):
    """Logs detection information to a specified file with JSON-compatible types."""
    log_data = []
    for det in detections:
        if det.track_id > 0:
            log_data.append({
                "frame_id": frame_id,
                "track_id": int(det.track_id),
                "class_id": int(det.det_class),
                "conf": float(det.conf),  # Convert to regular float
                "bbox": {
                    "left": float(det.bb_left),
                    "top": float(det.bb_top),
                    "width": float(det.bb_width),
                    "height": float(det.bb_height)
                },
                "mapped_coords": {
                    "x": float(det.y[0, 0]),  # Convert to float
                    "y": float(det.y[1, 0])
                }
            })
    with open(log_file, "a") as f:
        f.write(json.dumps(log_data) + "\n")

# Function to apply the ROI mask with green overlay
def apply_green_roi_mask(frame, points):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], (0, 255, 0))  # Green color
    alpha = 0.3  # Transparency
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

# Modify the main function to include the ROI mask
def main(args):
    yolo_version = "yolov8x"  # Update this based on the model version
    detector = Detector()
    unique_car_ids = set()

    class_list = [2, 5, 7]  # Define the classes to detect
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    


    video_out_path = f'output/output_{yolo_version}_conf{args.conf_thresh}_high{args.high_score}.mp4'
    log_file = f"output/detection_log_{yolo_version}_conf{args.conf_thresh}_high{args.high_score}.json"
    video_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Only create window if not running in headless mode
    if not args.headless:
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("demo", width, height)

    detector.load(args.cam_para, yolo_version)  # Pass yolo_version to Detector.load

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, None)

    log_file = "output/detection_log.json"
    open(log_file, "w").close()  # Clear log file at start

    frame_id = 1
    while True:
        ret, frame_img = cap.read()
        if not ret:
            break

        # Apply green ROI mask to the frame
        frame_img = apply_green_roi_mask(frame_img, roi_points)

        # Get detections and update the tracker
        dets = detector.get_dets(frame_img, args.conf_thresh, class_list)


        tracker.update(dets, frame_id)
        for det in dets:
            if det.track_id > 0:
                unique_car_ids.add(det.track_id)
        # Filter detections based on ROI
        for det in dets:
            # Check if detection's center point is inside the ROI
            center = (int(det.bb_left + det.bb_width / 2), int(det.bb_top + det.bb_height / 2))
            if cv2.pointPolygonTest(roi_points, center, False) >= 0:
                # Draw detection box if within ROI
                cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)),
                              (int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)),
                              (0, 255, 0), 2)
                cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Log detection data
        log_detection(dets, frame_id, log_file)
        frame_id += 1

        # Display the frame (if not in headless mode)
        if not args.headless:
            cv2.imshow("demo", frame_img)
            cv2.waitKey(1)

        # Write frame to output video
        video_out.write(frame_img)
    print(f"Total unique cars detected: {len(unique_car_ids)}")

    cap.release()
    video_out.release()
    if not args.headless:
        cv2.destroyAllWindows()

# Argument parser setup
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default="demo/new30s.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default="demo/cam_para.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=150.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=15.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.4, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.001, help='detection confidence threshold')
# parser.add_argument('--headless', type=bool, default=True, help='Run in headless mode without GUI display')
parser.add_argument('--headless', type=bool, default=False, help='Run in headless mode without GUI display')  # Set headless to False
args = parser.parse_args()

# Run main
main(args)


