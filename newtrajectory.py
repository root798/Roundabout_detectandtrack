import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tracker.ucmc import UCMCTrack
from tracker.kalman import KalmanTracker
from detector.mapper import Mapper
from collections import defaultdict
import argparse
import random
from matplotlib.colors import TABLEAU_COLORS
from itertools import cycle
# Given Homography Matrix (H)
H = np.array([
    [-2.19664313e-02,  7.91358229e-02,  4.00304124e+01],
    [ 4.56373888e-02, -1.64405670e-01, -8.31632638e+01],
    [-5.48766126e-04,  1.97697031e-03,  1.00000000e+00]
])
# Predefined colors for consistent and distinct colors
color_cycle = cycle(TABLEAU_COLORS.values())
color_map = defaultdict(lambda: next(color_cycle))  # Assign a new color if not in the map

def map_image_to_real_world(homography_matrix, image_point):
    """
    Map a single image pixel coordinate to the real-world coordinate using the homography matrix.

    :param homography_matrix: 3x3 homography matrix.
    :param image_point: Pixel coordinate as (x, y).
    :return: Real-world coordinate as (lat, lng).
    """
    # Convert the image point to homogeneous coordinates
    image_point_homogeneous = np.array([image_point[0], image_point[1], 1], dtype=np.float32)
    # Map the point using the homography matrix
    real_world_point = np.dot(homography_matrix, image_point_homogeneous)
    # Normalize to get (lat, lng)
    real_world_point /= real_world_point[2]
    return real_world_point[0], real_world_point[1]
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
        results = self.model(frame, imgsz=2560)
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
            # det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height]) 
            # We'll now rely on the homography for mapping, not the mapper.
            det_id += 1
            dets.append(det)

        return dets

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
    class_labels = {2: "Car", 5: "Bus", 7: "Truck"}

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
        ax.set_title("Trajectories of Tracked Vehicles (Real-World Coordinates)")
        ax.set_xlabel("X (Real-World)")
        ax.set_ylabel("Y (Real-World)")

        for track_id, data in trajectory_data.items():
            if len(data['coords']) > 0:  # Plot only if there are points
                x_vals, y_vals = zip(*data['coords'])
                ax.scatter(x_vals, y_vals, label=f'Track {track_id}', color=color_map[track_id])

        # Optionally, add a legend if needed
        # if trajectory_data:
        #     ax.legend()
            
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
            center_pixel = (int(det.bb_left + det.bb_width / 2), int(det.bb_top + det.bb_height / 2))

            # Check if detection is inside the ROI
            if cv2.pointPolygonTest(roi_points, center_pixel, False) >= 0:
                
                if det.det_class == 2:  # Car
                    unique_car_ids.add(det.track_id)
                    current_cars_in_intersection += 1
                elif det.det_class == 7:  # Truck
                    unique_truck_ids.add(det.track_id)
                    current_trucks_in_intersection += 1

                if frame_id > 5:
                    # Map pixel center to real-world coordinates
                    #print(center_pixel, "c")
                    real_world_point = map_image_to_real_world(H, center_pixel)
                    #print(real_world_point,"r")
                    # det.y = np.array([[real_world_point[0]], [real_world_point[1]]])
                    # det.R = np.eye(2) * 1.0  # Or a suitable covariance for your measurements
                    trajectory_data[det.track_id]['coords'].append(real_world_point)
                    trajectory_data[det.track_id]['class_id'] = det.det_class

                # Define the class name and tracking ID
                class_name = class_labels.get(det.det_class, "Unknown")
                label = f"{class_name} ID:{det.track_id}"

                # Draw the bounding box, classification, and ID
                cv2.rectangle(
                    frame_resized,
                    (int(det.bb_left // 2), int(det.bb_top // 2)),
                    (int((det.bb_left + det.bb_width) // 2), int((det.bb_top + det.bb_height) // 2)),
                    (0, 255, 0), 2
                )
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
            cv2.putText(canvas, line, (car_count_position[0], y_position + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Show the final canvas
        cv2.imshow("Real-Time Tracking and Analysis", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    plt.close()

# Argument parser setup
parser = argparse.ArgumentParser(description='Real-time Tracking and Trajectory Plotting')
parser.add_argument('--video', type=str, default="C:/research/Archive/Untitled.mp4", help='video file name') #C:/research/new30s.mp4
parser.add_argument('--cam_para', type=str, default="C:/research/UCMCTrack/demo/cam_para.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=20, help='wx')
parser.add_argument('--wy', type=float, default=20, help='wy')
parser.add_argument('--vmax', type=float, default=20, help='vmax')
parser.add_argument('--a', type=float, default=50.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=2.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.2, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
parser.add_argument('--fps', type=float, default=30, help='frames per second')
args = parser.parse_args()

# Run the main function
main(args)