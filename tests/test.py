import cv2
import numpy as np

cam = cv2.VideoCapture(-1)
if not cam.isOpened():
    print("camera could not initialise.")
    exit()
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame.")
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowRed = np.array([0, 150, 140])
    highRed = np.array([10, 220, 255])
    mask = cv2.inRange(hsv, lowRed, highRed)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, [100, 200, 0], 4)
    cv2.imshow('Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()