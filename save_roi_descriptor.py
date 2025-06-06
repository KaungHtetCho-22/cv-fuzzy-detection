import cv2
import numpy as np
import os

video_path = "videos/people_walking.mp4"
video = cv2.VideoCapture(video_path)
ret, frame = video.read()

scale = 1
x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

def coordinat_chooser(event, x, y, flags, param):
    global x_min, y_min, x_max, y_max

    orig_x = int(x / scale)
    orig_y = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        x_min = min(orig_x, x_min)
        y_min = min(orig_y, y_min)
        x_max = max(orig_x, x_max)
        y_max = max(orig_y, y_max)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    if event == cv2.EVENT_MBUTTONDOWN:
        print("Reset coordinate data")
        x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

cv2.namedWindow('coordinate_screen')
cv2.setMouseCallback('coordinate_screen', coordinat_chooser)

while True:
    display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.imshow("coordinate_screen", display_frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()

# Extract ROI
roi_image = frame[y_min+2:y_max-2, x_min+2:x_max-2]
roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

# FAST + BRIEF
fast = cv2.FastFeatureDetector_create(threshold=1)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

keypoints = fast.detect(roi_gray, None)
keypoints, descriptors = brief.compute(roi_gray, keypoints)

if descriptors is None or len(descriptors) == 0:
    print("No descriptors extracted. Try selecting a clearer ROI.")
    exit()

# Save to disk
os.makedirs("saved_features", exist_ok=True)
np.savez("saved_features/roi_features_people.npz", descriptors=descriptors)
print("Saved descriptors to saved_features/roi_features_people.npz")
