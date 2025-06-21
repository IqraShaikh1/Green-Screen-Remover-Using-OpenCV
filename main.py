# Importing libraries
import cv2
import numpy as np

# Reading the green screen and background image
green_screen_img = cv2.imread("mango.jpg")
bg_img = cv2.imread("aesthetic_bg.jpg")
# Resize the background image to match the green screen image dimensions
bg_img_resized = cv2.resize(bg_img, (green_screen_img.shape[1], green_screen_img.shape[0]))

# Convert the green screen image to HSV color space
hsv_img = cv2.cvtColor(green_screen_img, cv2.COLOR_BGR2HSV)

# Define the green color range in HSV (tighter range to avoid false detection)
lower_green = np.array([100, 100, 50])  # Lower bound for green
upper_green = np.array([140, 255, 255])  # Upper bound for green

# Create a mask for the green color
mask = cv2.inRange(hsv_img, lower_green, upper_green)

#Additional filtering to exclude black/gray regions (by brightness)
# Convert the image to grayscale and create a mask for low-brightness pixels
gray_img = cv2.cvtColor(green_screen_img, cv2.COLOR_BGR2GRAY)
black_mask = cv2.inRange(gray_img, 0, 50)  # Exclude very dark pixels (range can be adjusted)

# Combine the green mask and the black exclusion mask
final_mask = cv2.bitwise_and(mask, cv2.bitwise_not(black_mask))

# Apply Gaussian blur to smooth out the mask (optional)

# Invert the mask to get the non-green areas
mask_inv = cv2.bitwise_not(final_mask)

# Extract the foreground (non-green areas) from the green screen image
foreground = cv2.bitwise_and(green_screen_img, green_screen_img, mask=mask_inv)

# Extract the background where the green screen was (using the mask on the background image)
background = cv2.bitwise_and(bg_img_resized, bg_img_resized, mask=final_mask)

# Combine the foreground and the background
output_img = cv2.add(foreground, background)

# Show the result
cv2.imshow("Output Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
