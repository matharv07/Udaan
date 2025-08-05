import cv2
import numpy as np
# opencv represents images as width x height -> LHC is 0,0 -> image size is 480 x 640

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
start = (200 , 260)
end = (280, 380)
color = (100, 200, 0)
thickness = 4
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowBlue = np.array([0, 0, 0])
    highBlue = np.array([255, 255, 255])
    mask = cv2.inRange(frame, lowBlue, highBlue)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    th = cv2.threshold(lab[:, :, 2], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #thresh: 127, #maxval: 255
    res = cv2.bitwise_and(frame, frame, mask = mask)
    frame = cv2.rectangle(frame, start, end, color, thickness)
    print("frame color: ", frame[240, 320], "hsv color: ", frameHSV[240,320])
    cv2.imshow('HSV Feed', frameHSV)
    cv2.imshow('Feed', frame)
    cv2.imshow('Detected Feed', mask)
    cv2.imshow('Bitwise Feed', res)
    cv2.imshow('LAB', lab)
    cv2.imshow('thresh', th)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()