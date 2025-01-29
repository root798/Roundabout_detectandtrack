import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tracker.ucmc import UCMCTrack
from tracker.kalman import KalmanTracker
from detector.mapper import Mapper
from collections import defaultdict
import argparse

# Define the ROI points as a numpy array
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
# roi_points = np.array([
#     [1065, 174], [1047, 187], [1017, 206], [988, 231], [969, 243],
#     [929, 284], [895, 307], [816, 327], [743, 334], [647, 341],
#     [588, 332], [521, 328], [466, 318], [380, 311], [298, 303],
#     [239, 297], [186, 287], [137, 287], [73, 335], [124, 341],
#     [160, 344], [208, 352], [259, 359], [295, 359], [348, 368],
#     [372, 372], [425, 378], [469, 386], [507, 400], [543, 420],
#     [573, 446], [611, 485], [604, 528], [573, 597], [539, 641],
#     [497, 697], [449, 744], [397, 790], [363, 827], [269, 922],
#     [184, 1004], [108, 1075], [235, 1078], [414, 1079], [550, 1078],
#     [579, 1025], [608, 967], [632, 923], [657, 874], [690, 833],
#     [722, 786], [759, 731], [786, 687], [850, 644], [895, 619],
#     [953, 590], [1014, 571], [1089, 556], [1190, 569], [1265, 573],
#     [1332, 579], [1392, 590], [1447, 596], [1540, 612], [1631, 632],
#     [1679, 645], [1775, 658], [1841, 691], [1890, 702], [1895, 584],
#     [1832, 569], [1574, 521], [1448, 493], [1298, 455], [1202, 414],
#     [1151, 389], [1099, 356], [1083, 309], [1093, 258], [1107, 227],
#     [1122, 192], [1137, 172], [1146, 152], [1160, 136], [1141, 129],
#     [1117, 127], [1097, 129], [1070, 125], [1070, 143]
# ], np.int32)
# Define Detection class
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

# Define Detector class
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
        results = self.model(frame, imgsz=2560)#1920
        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
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

# Apply green ROI mask function
def apply_green_roi_mask(frame, points, original_size, resized_size):
    scale_x = resized_size[0] / original_size[0]
    scale_y = resized_size[1] / original_size[1]
    scaled_points = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in points], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [scaled_points], (0, 255, 0))  # Green color for ROI
    alpha = 0.3  # Transparency level
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
def main(args):
    yolo_version = "yolo11x"
    detector = Detector()
    detector.load(args.cam_para, yolo_version)
    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, args.fps, "MOT", args.high_score, False, None)
    unique_car_ids = set()
    unique_truck_ids = set()
    trajectory_data = defaultdict(lambda: {'coords': [], 'class_id': None})

    # Define class labels
    class_labels = {2: "Car", 5: "Bus", 7: "Truck"}  # Define labels for known classes

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_display_width = width // 2
    video_display_height = height // 2
    canvas_width = video_display_width + 400
    canvas_height = video_display_height + 10
    car_count_position = (video_display_width + 30, canvas_height - 100)

    fig, ax = plt.subplots()
    plt.ion()

    def update_trajectory_graph():
        ax.clear()
        ax.set_title("Trajectories of Tracked Vehicles")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        for track_id, data in trajectory_data.items():
            if len(data['coords']) > 1:#data['coords']:
                x_vals, y_vals = zip(*data['coords'])
                x_vals = [-x for x in x_vals]
                y_vals = [-y for y in y_vals]
                ax.scatter(x_vals, y_vals, marker="o", label=f'Track {track_id}')#ax.plot(x_vals, y_vals, marker="o", label=f'Track {track_id}')
        fig.canvas.draw()
        fig_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        fig_image = fig_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        fig_image = cv2.cvtColor(fig_image, cv2.COLOR_RGBA2RGB)
        return fig_image

    frame_id = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (video_display_width, video_display_height))
        original_size = (width, height)
        resized_size = (video_display_width, video_display_height)
        frame_resized = apply_green_roi_mask(frame_resized, roi_points, original_size, resized_size)

        dets = detector.get_dets(frame, args.conf_thresh, [2, 5, 7])
        tracker.update(dets, frame_id)

        # Reset intersection counts for each frame
        current_cars_in_intersection = 0
        current_trucks_in_intersection = 0

        for det in dets:
            center = (int(det.bb_left + det.bb_width / 2), int(det.bb_top + det.bb_height / 2))

            # Check if detection is inside the ROI
            if cv2.pointPolygonTest(roi_points, center, False) >= 0:
                if det.det_class == 2:  # Car
                    unique_car_ids.add(det.track_id)
                    current_cars_in_intersection += 1
                elif det.det_class == 7:  # Truck
                    #unique_truck_ids.add(det.track_id)
                    current_trucks_in_intersection += 1
                if frame_id > 5:
                    trajectory_data[det.track_id]['coords'].append((det.y[0, 0], det.y[1, 0]))

                # Define the class name, tracking ID, and confidence score
                class_name = class_labels.get(det.det_class, "Unknown")
                class_name ="car"
                label = f"{class_name} ID:{det.track_id} "#{det.conf:.2f}"

                # Draw the bounding box, classification, confidence score, and tracking ID
                cv2.rectangle(
                    frame_resized,
                    (int(det.bb_left // 2), int(det.bb_top // 2)),
                    (int((det.bb_left + det.bb_width) // 2), int((det.bb_top + det.bb_height) // 2)),
                    (0, 255, 0), 2
                )
                # Display the label with classification, confidence, and ID above the bounding box
                cv2.putText(
                    frame_resized, label,
                    (int(det.bb_left // 2), int((det.bb_top // 2) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                )

        # Prepare the display canvas and add elements
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        canvas[0:video_display_height, 0:video_display_width] = frame_resized

        # Add trajectory graph
        trajectory_img = update_trajectory_graph()
        traj_height, traj_width, _ = trajectory_img.shape
        resized_traj_img = cv2.resize(trajectory_img, (390, int(traj_height * (390 / traj_width))))
        canvas[10:10+resized_traj_img.shape[0], video_display_width+10:video_display_width+10+resized_traj_img.shape[1]] = resized_traj_img

        # Display total and current counts for cars and trucks
        count_text = (
            f"Total Cars: {len(unique_car_ids)}\n"
            f"Cars in Intersection: {current_cars_in_intersection}\n"
            f"Total Trucks: {len(unique_truck_ids)}\n"
            f"Trucks in Intersection: {current_trucks_in_intersection}"
        )

        # Display the count text on the canvas
        y_position = car_count_position[1]
        for i, line in enumerate(count_text.split('\n')):
            cv2.putText(canvas, line, (car_count_position[0], y_position + i * 25),  # Adjust line spacing with *25
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Show the final canvas
        cv2.imshow("Real-Time Tracking and Analysis", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    plt.close()

# # Argument parser setup
# parser = argparse.ArgumentParser(description='Real-time Tracking and Trajectory Plotting')
# parser.add_argument('--video', type=str, default="C:/research/new30s.mp4", help='video file name')
# parser.add_argument('--cam_para', type=str, default="C:/research/UCMCTrack/demo/cam_para.txt", help='camera parameter file name')
# parser.add_argument('--wx', type=float, default=10, help='wx')
# parser.add_argument('--wy', type=float, default=10, help='wy')
# parser.add_argument('--vmax', type=float, default=12, help='vmax')
# parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
# parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
# parser.add_argument('--high_score', type=float, default=0.2, help='high score threshold')
# parser.add_argument('--conf_thresh', type=float, default=0.00001, help='detection confidence threshold')
# parser.add_argument('--fps', type=float, default=30, help='frames per second')
# args = parser.parse_args()

# Argument parser setup
parser = argparse.ArgumentParser(description='Real-time Tracking and Trajectory Plotting')
parser.add_argument('--video', type=str, default="C:/research/Archive/Untitled.mp4", help='video file name') #C:/research/new30s.mp4
parser.add_argument('--cam_para', type=str, default="C:/research/UCMCTrack/demo/cam_para.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=10, help='wx')
parser.add_argument('--wy', type=float, default=10, help='wy')
parser.add_argument('--vmax', type=float, default=15, help='vmax')
parser.add_argument('--a', type=float, default=30.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=0.1, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.3, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.3, help='detection confidence threshold')
parser.add_argument('--fps', type=float, default=30, help='frames per second')
args = parser.parse_args()
# Run the main function
main(args)
