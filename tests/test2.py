from traceback import print_tb

import cv2
import numpy as np

cap = cv2.VideoCapture(-1)
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
    #frame = cv2.rectangle(frame, start, end, color, thickness)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    detect = cv2.bitwise_and(frameHSV, frameHSV, mask = thresh)
    #detect = cv2.blur(detect, (5,5))
    #mask = cv2.inRange(detect, lowRed, highRed)
    mask = cv2.inRange(frameHSV, lowRed, highRed)
    canny = cv2.Canny(gray, 150, 175)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    start, end, check = [480, 640], [0, 0], False
    del_arr = []
    for contour in contours:
        check = True
        if contour[0][0][0] < start[0]:
            start[0] = contour[0][0][0]
        if contour[0][0][1] < start[1]:
            start[1] = contour[0][0][1]
        if contour[0][0][0] > end[0]:
            end[0] = contour[0][0][0]
        if contour[0][0][1] > end[1]:
            end[1] = contour[0][0][1]
    if not check:
        print(start, end)
        s = (start[0], start[1])
        e = (end[0], end[1])
        c = (100, 200, 0)
        t = 4
        frame = cv2.rectangle(frame, s, e, c, t)
    else:
        print(check)
        frame = cv2.drawContours(frame, contours, -1, [100, 200, 0], 4)
    print("frame color: ", frame[240, 320], "hsv color: ", frameHSV[240,320])
    cv2.imshow('HSV Feed', frameHSV)
    cv2.imshow('Detected Feed', mask)
    cv2.imshow('Bitwise Feed', res)
    cv2.imshow('Feed', frame)
    cv2.imshow('Binary image', thresh)
    cv2.imshow('detect', detect)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()