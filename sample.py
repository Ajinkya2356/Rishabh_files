import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

# Paths to the images
master_image_path = "./clear_section.png"
input_image_path = "./Analog_mobile/test/class 0/bad_m_46.png"

# Load images
master_image = cv2.imread(master_image_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector and matcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# Detect and match keypoints
master_kp, master_desc = sift.detectAndCompute(master_image, None)
input_kp, input_desc = sift.detectAndCompute(input_image, None)
matches = bf.knnMatch(input_desc, master_desc, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Compute the homography matrix
if len(good_matches) > 10:
    src_pts = np.float32([input_kp[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([master_kp[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Align the input image
    aligned_input = cv2.warpPerspective(
        input_image, H, (master_image.shape[1], master_image.shape[0])
    )
else:
    raise ValueError("Not enough good matches to compute homography!")

# Compute SSIM and difference image
(score, diff) = compare_ssim(master_image, aligned_input, full=True)
diff = (diff * 255).astype("uint8")

# Threshold the difference image
_, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)

# Morphological operations for cleaning
kernel = np.ones((5, 5), np.uint8)
diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)

# Find contours of defective regions
contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Highlight defective segments
output_image = cv2.cvtColor(master_image, cv2.COLOR_GRAY2BGR)
for contour in contours:
    if cv2.contourArea(contour) < 10:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display results
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title("Aligned Input Image")
plt.imshow(cv2.cvtColor(aligned_input, cv2.COLOR_GRAY2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Difference Image (Thresholded)")
plt.imshow(diff_thresh, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Defective Segments Highlighted")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
