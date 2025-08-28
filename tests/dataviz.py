import cv2
import numpy as np

cap = cv2.VideoCapture(-1)
i = 0
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowRed = np.array([0, 130, 130])
    highRed = np.array([60, 255, 255])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100])
    maskOR = cv2.inRange(hsv_image, lower_black, upper_black)
    checker = frame.copy()
    checker[maskOR > 0] = (40, 60, 180)
    r, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    detect = cv2.bitwise_and(frameHSV, frameHSV, mask=thresh)
    cone = cv2.bitwise_and(frame, frame, mask=thresh)
    mask = cv2.inRange(detect, lowRed, highRed)
    mask = cv2.medianBlur(mask, 5)
    canny = cv2.Canny(gray, 150, 175)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = frame.copy()
    contour = contour[0] if len(contour) == 2 else contour[1]
    x, y, w, h, area = 0, 0, 0, 0, 0
    for cntr in contour:
        xl, yl, wl, hl = cv2.boundingRect(cntr)
        cv2.rectangle(result, (xl, yl), (xl + wl, yl + hl), (255, 0, 0), 2)
    cv2.imshow('checker', checker)
    cv2.imshow('Feed', frame)
    cv2.imshow('Binary image', thresh)
    cv2.imshow('bounds', result)
    cv2.imshow('detect', detect)
    cv2.imshow('fore', cone)
    cv2.imshow('Bitwise Feed', res)
    cv2.imshow('canny', canny)
    index = "img" + str(i) + ".jpg"
    cv2.imwrite(index, res)
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()