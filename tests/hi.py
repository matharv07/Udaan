import cv2

shot = cv2.imread("/home/atharv/Pictures/poster1.png", cv2.IMREAD_COLOR)
blue, green, red = cv2.split(shot)
cv2.imshow("shot", shot)
print(shot.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()