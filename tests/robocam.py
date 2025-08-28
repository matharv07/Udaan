import cv2
import numpy as np

# 1. Load the image
image = cv2.imread('/home/atharv/PycharmProjects/Udaan/tests/genphoto.jpg') # Replace with your image path

# 2. Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. Define black color range in HSV
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 60]) # Adjust 'V' (Value) if needed for very dark grays

# 4. Create a mask for black pixels
mask = cv2.inRange(hsv_image, lower_black, upper_black)

# 5. Define orange color in BGR
orange_color_bgr = (20, 165, 200) # BGR for a common orange shade

# 6. Change color
result_image = image.copy()
result_image[mask > 0] = orange_color_bgr

# 7. Display or Save the Result
cv2.imshow('Original Image', image)
cv2.imshow('Orange Modified Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()