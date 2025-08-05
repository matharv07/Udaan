import cv2
import numpy as np

def find_common_contours(contours1, contours2, image_shape):
    """
    Finds the common contours between two contour lists by checking for intersection.

    Args:
        contours1 (list): The first list of contours.
        contours2 (list): The second list of contours.
        image_shape (tuple): The shape of the original image (height, width).

    Returns:
        list: A list of contours representing the common lines.
    """
    # Create two blank masks to draw the contours on
    # The masks should be single-channel (grayscale)
    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)

    # Draw and fill the first set of contours on mask1 in white (255)
    # -1 indicates to draw all contours, cv2.FILLED fills the contours
    cv2.drawContours(mask1, contours1, -1, 255, thickness=cv2.FILLED)

    # Draw and fill the second set of contours on mask2 in white (255)
    cv2.drawContours(mask2, contours2, -1, 255, thickness=cv2.FILLED)

    # Find the intersection using a bitwise AND operation
    # This will result in white pixels only where both masks had white pixels
    intersection_mask = cv2.bitwise_and(mask1, mask2)

    # Find the contours on the intersection mask
    # RETR_EXTERNAL retrieves only the extreme outer contours
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
    common_contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return common_contours

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Create some dummy contours for demonstration
    #    In a real application, you would get these from findContours() on two different images
    image_width, image_height = 500, 500
    # Image shape for mask creation (height, width)
    image_shape = (image_height, image_width)

    # Contour List 1: A square and a circle
    # Contours are typically lists of numpy arrays of points
    contour_list1 = [
        np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.int32).reshape((-1, 1, 2)), # A square
        cv2.ellipse2Poly((300, 300), (50, 50), 0, 0, 360, 10).reshape((-1, 1, 2))         # A circle
    ]

    # Contour List 2: A larger square that overlaps with the first, and the same circle
    contour_list2 = [
        np.array([[150, 150], [250, 150], [250, 250], [150, 250]], dtype=np.int32).reshape((-1, 1, 2)), # An overlapping square
        cv2.ellipse2Poly((300, 300), (50, 50), 0, 0, 360, 10).reshape((-1, 1, 2))         # The same circle
    ]

    # 2. Find the common contours
    print("Finding common contours...")
    common_contours = find_common_contours(contour_list1, contour_list2, image_shape)
    print(f"Found {len(common_contours)} common contours.")

    # 3. Display the results
    #    Create a blank image to draw the common contours on
    #    It should be a 3-channel (BGR) image for color drawing
    output_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Draw the common contours in a distinct color (e.g., green) with a thickness of 2
    cv2.drawContours(output_image, common_contours, -1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Common Contours", output_image)
    print("Displaying common contours. Press any key to close the window.")
    cv2.waitKey(0) # Waits indefinitely for a key press
    cv2.destroyAllWindows() # Closes all OpenCV windows
