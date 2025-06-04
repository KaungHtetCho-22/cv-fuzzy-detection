import cv2

def draw_boxes(img_path, boxes, out_path):
    img = cv2.imread(img_path)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
