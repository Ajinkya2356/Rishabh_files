import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load the captured image and reference (master) image
captured_image = cv2.imread("./generated_images/train/class 0/bad_3.png", cv2.IMREAD_GRAYSCALE)
master_image = cv2.imread("./generated_images/train/class 1/good_3.png", cv2.IMREAD_GRAYSCALE)

# Resize images to the same size if necessary
captured_image = cv2.resize(captured_image, (85, 226))  # Adjust size
master_image = cv2.resize(master_image, (85, 226))

# Preprocess: Apply thresholding to detect "on" segments
_, captured_thresh = cv2.threshold(captured_image, 128, 255, cv2.THRESH_BINARY)
_, master_thresh = cv2.threshold(master_image, 128, 255, cv2.THRESH_BINARY)

# Subtract master image from captured image to find differences
difference = cv2.absdiff(master_thresh, captured_thresh)

# Count the number of non-zero pixels (differences)
difference_count = cv2.countNonZero(difference)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.title("Captured Image"), plt.imshow(captured_image, cmap="gray")
plt.subplot(1, 3, 2), plt.title("Master Image"), plt.imshow(master_image, cmap="gray")
plt.subplot(1, 3, 3), plt.title("Difference"), plt.imshow(difference, cmap="gray")
plt.show()

# Determine if the image matches the master image
if difference_count == 0:
    print("✅ All segments are ON and match the master image.")
else:
    print(f"❌ Segments do not match. {difference_count} differing pixels found.")
