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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.array(np.absolute(lap))
    canny = cv2.Canny(gray, 150, 175)
    top = cv2.bitwise_not(frame, frame, mask = canny)
    contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #gray = cv2.drawContours(gray, contours, -1, (100, 200, 0), 4)
    cv2.imshow('Feed', frame)
    cv2.imshow('Lap', lap)
    cv2.imshow('canny', canny)
    cv2.imshow('top', top)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()