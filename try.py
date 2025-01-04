import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to the images
master_image_path = "./clear_section.png"
captured_image_path = "./Analog_mobile/test/class 0/bad_m_46.png"

# Load images in grayscale
master_image = cv2.imread(master_image_path, cv2.IMREAD_GRAYSCALE)
captured_image = cv2.imread(captured_image_path, cv2.IMREAD_GRAYSCALE)

# Resize captured image to match master image dimensions
captured_image = cv2.resize(captured_image, (master_image.shape[1], master_image.shape[0]))

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect and compute features
master_kp, master_desc = orb.detectAndCompute(master_image, None)
captured_kp, captured_desc = orb.detectAndCompute(captured_image, None)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(master_desc, captured_desc)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Identify missing segments
missing_segments = []
for i, match in enumerate(matches):
    if match.distance > 50:  # Adjust the distance threshold as needed
        master_idx = match.queryIdx
        captured_idx = match.trainIdx
        
        # Get the keypoint locations
        master_x, master_y = master_kp[master_idx].pt
        captured_x, captured_y = captured_kp[captured_idx].pt
        
        # Check if the keypoints are in a similar location
        if abs(master_x - captured_x) > 20 or abs(master_y - captured_y) > 20:
            missing_segments.append(cv2.KeyPoint(master_x, master_y, 1))

# Highlight missing segments
output_image = cv2.cvtColor(master_image, cv2.COLOR_GRAY2BGR)
for segment in missing_segments:
    cv2.circle(output_image, (int(segment.pt[0]), int(segment.pt[1])), 5, (0, 0, 255), -1)

# Visualize results
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title("Master Image")
plt.imshow(master_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Captured Image")
plt.imshow(captured_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Missing Segments Highlighted")
plt.imshow(output_image)
plt.axis("off")

plt.tight_layout()
plt.show()