import numpy as np
import cv2

# Define the updated list of selected points for the ROI polygon
roi_points = np.array([
    [28, 1046], [109, 977], [269, 846], [426, 715], [507, 614],
    [570, 520], [521, 456], [427, 411], [308, 392], [182, 372],
    [39, 350], [196, 251], [338, 273], [518, 299], [663, 315],
    [805, 319], [920, 292], [1045, 236], [1099, 195], [1155, 154],
    [1243, 103], [1311, 114], [1256, 176], [1205, 249], [1198, 346],
    [1286, 433], [1454, 500], [1634, 555], [1786, 595], [1902, 621],
    [1888, 797], [1665, 738], [1408, 684], [1184, 650], [1012, 680],
    [856, 774], [739, 921], [659, 1047]
], np.int32)

# Function to apply a semi-transparent green ROI mask
def apply_green_roi_mask(frame, points):
    # Create an overlay the same size as the frame
    overlay = frame.copy()
    # Fill the polygon on the overlay with green color
    cv2.fillPoly(overlay, [points], (0, 255, 0))
    # Blend the overlay with the original frame (adjust the transparency with alpha)
    alpha = 0.3  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

# Example of reading and processing frames from a video
cap = cv2.VideoCapture("C:\\Users\\17264\\OneDrive\\桌面\\jhu\\sustainbale and efficient computing for transportation\\new30s.mp4")  # Replace with your video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply the green ROI mask to the current frame
    masked_frame = apply_green_roi_mask(frame, roi_points)

    # Display the masked frame (optional, or save/write the frame to an output)
    cv2.imshow("Masked Frame with Green Overlay", masked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
