from imageio.v2 import sizes
from skimage import feature, color, io
import matplotlib.pyplot as plt
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowRed = np.array([0, 150, 140])
    highRed = np.array([10, 220, 255])
    mask = cv2.inRange(frameHSV, lowRed, highRed)
    contours, x = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    raw_dist = np.empty(mask.shape, dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if len(contours) > 0:
                raw_dist[i, j] = cv2.pointPolygonTest(contours[0], (j, i), True)
    cv2.boundingRect()
    minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
    minVal = abs(minVal)
    maxVal = abs(maxVal)

    # Depicting the  distances graphically
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if raw_dist[i, j] < 0:
                drawing[i, j, 0] = 255 - abs(raw_dist[i, j]) * 255 / minVal
            elif raw_dist[i, j] > 0:
                drawing[i, j, 2] = 255 - raw_dist[i, j] * 255 / maxVal
            else:
                drawing[i, j, 0] = 255
                drawing[i, j, 1] = 255
                drawing[i, j, 2] = 255

    cv2.circle(drawing, maxDistPt, int(maxVal), (255, 255, 255), 1, cv2.LINE_8, 0)
    cv2.imshow('Source', mask)
    cv2.imshow('Distance and inscribed circle', drawing)
    cv2.imshow('frame', frame)
    cv2.imshow('detect', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()