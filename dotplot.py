import cv2
import numpy as np

# Load the first frame of the video
video_path = "C:/research/Archive/Untitled.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Function to capture mouse click events
roi_points = []
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Frame", frame)
        print(f"Point selected: ({x}, {y})")

# Set up window and mouse callback
cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", select_points)

print("Click on points around the cross-shaped intersection. Press 'q' when done.")

# Wait until 'q' key is pressed to close the window
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Convert to numpy array for OpenCV
roi_polygon = np.array(roi_points, np.int32)

# Print the selected ROI points
print("Selected ROI points:", roi_polygon)
