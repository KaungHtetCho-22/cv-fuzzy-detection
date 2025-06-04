from detector.detect import run_detection
from fuzzy.fuzzy_filter import filter_detection
from utils.visualize import draw_boxes
import os
# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    image_path = 'data/sample.jpg'
    results = run_detection(image_path)

    filtered_detections = []
    for box in results.boxes:
        conf = float(box.conf)
        iou = 1.0 
        trust = filter_detection(conf, iou)

        if trust > 0.5:
            filtered_detections.append(box)

    draw_boxes(image_path, filtered_detections, out_path='results/filtered.jpg')

if __name__ == "__main__":
    main()
