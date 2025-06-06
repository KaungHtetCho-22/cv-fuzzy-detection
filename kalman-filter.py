import cv2
import numpy as np

video_path = "videos/people_walking.mp4"
scale = 1 # scale only for display

# Load descriptors
data = np.load("saved_features/roi_features_people.npz")
descriptors_1 = data["descriptors"]

fast = cv2.FastFeatureDetector_create(threshold=1)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
bf = cv2.BFMatcher()

# Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

def detect_target_fast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kps = fast.detect(gray, None)
    kps, desc = brief.compute(gray, kps)
    if desc is not None:
        matches = bf.match(descriptors_1, desc)
        if matches:
            pts = [kps[m.trainIdx].pt for m in matches]
            avg_x = sum(p[0] for p in pts) / len(pts)
            avg_y = sum(p[1] for p in pts) / len(pts)
            return int(avg_x), int(avg_y)
    return None

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict with Kalman
    
    predicted = kalman.predict()
    predicted_x = predicted[0].item()
    predicted_y = predicted[1].item()
    predicted_dx = predicted[2].item()
    predicted_dy = predicted[3].item()

    print(f"Predicted velocity: dx={predicted_dx:.2f}, dy={predicted_dy:.2f}")


    ball_position = detect_target_fast(frame)

    if ball_position:
        measured_x, measured_y = ball_position
        kalman.correct(np.array([[np.float32(measured_x)], [np.float32(measured_y)]]))
        # print(f"[TRACKED] Measured at: ({measured_x}, {measured_y})")
    else:
        print("[WARNING] No match found")

    display_frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # Draw Kalman predicted position (red)
    cv2.circle(display_frame, (int(predicted_x * scale), int(predicted_y * scale)), 8, (0, 0, 255), 2)

    # Draw measured position (green)
    if ball_position:
        cv2.circle(display_frame, (int(measured_x * scale), int(measured_y * scale)), 6, (0, 255, 0), 2)

    cv2.imshow("Kalman Ball Tracking", display_frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
