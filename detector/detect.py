from ultralytics import YOLO
import cv2

def run_detection(image_path, model_path='models/yolov8n.pt'):
    model = YOLO(model_path)
    results = model(image_path)[0]
    return results
