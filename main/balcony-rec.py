import cv2
import os
import datetime

OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(OUTPUT_DIR, f"webcam_photo_{timestamp}.jpg")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print("Success")
    else:
        print("Error")
cap.release()