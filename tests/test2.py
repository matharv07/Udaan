import cv2
import numpy as np

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowRed = np.array([0, 100, 100])
    highRed = np.array([10, 220, 255])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    detect = cv2.bitwise_and(frameHSV, frameHSV, mask = thresh)
    mask = cv2.inRange(detect, lowRed, highRed)
    mask = cv2.medianBlur(mask,5)
    canny = cv2.Canny(gray, 150, 175)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    result = frame.copy()
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0] if len(contour) == 2 else contour[1]
    x, y, w, h, area = 0, 0, 0, 0, 0
    for cntr in contour:
        a = cv2.contourArea(cntr)
        if a > area:
            area = a
            x, y, w, h = cv2.boundingRect(cntr)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(area)
    str = ""
    if x + w/2 > 330:
        str += "left, "
    elif x + w/2 < 310:
        str += "right, "
    else:
        str += "horz ok, "
    if y + h/2 > 250:
        str += "down"
    elif y + h/2 < 230:
        str += "up"
    else:
        str += "vert ok"
    print(str)
    factor = (62752.714285714 - 33949.357142857)/6.0
    a = (62752.714285714 - area)/factor + 25
    print(a)
    cv2.imshow("bounding_box", result)
    #print("frame color: ", frame[240, 320], "hsv color: ", frameHSV[240,320])
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

""" I'll add comments later """