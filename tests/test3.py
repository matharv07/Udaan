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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    detect = cv2.bitwise_and(frameHSV, frameHSV, mask = thresh)
    #detect = cv2.blur(detect, (5,5))
    mask = cv2.inRange(frameHSV, lowRed, highRed)
    canny = cv2.Canny(gray, 130, 175)
    r, cancontours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #frame = cv2.drawContours(frame, cancontours, -1, (100, 200, 0), 4)
    cv2.imshow("contour", canny)
    cv2.imshow("feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()