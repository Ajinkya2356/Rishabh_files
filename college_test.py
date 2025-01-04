from pyueye import ueye
import numpy as np
import cv2
import os
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt


# Define the desired width and height for the live video preview
desired_width = 470  # Set the desired width here
desired_height = 420  # Set the desired height here

# Define the directory to save captured images
save_directory = "./section 2 clear"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

esrgan_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

# Define the predefined sections (rectangles) to capture
# Each tuple represents (x_start, y_start, x_end, y_end)
predefined_sections = [(45, 9, 457, 408)]

crop_areas_percentage = [
    (0.01, 0.09, 0.27, 0.83),
    (0.24, 0.08, 0.83, 0.83),
    (0.01, 0.80, 0.8, 1),
    (0.8, 0.09, 1, 1),
    (0.24, 0, 1, 0.1),
]

# Define camera settings
hCam = ueye.HIDS(0)  # Camera handle (0 for default camera)
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
rectAOI = ueye.IS_RECT()

# Initialize the camera
ueye.is_InitCamera(hCam, None)

# Get camera information
ueye.is_GetCameraInfo(hCam, cInfo)
ueye.is_GetSensorInfo(hCam, sInfo)

# Set color mode to RGB8
ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)

# Set the area of interest (AOI)
ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))

# Convert c_int to Python int for actual width and height from the camera
camera_width = int(rectAOI.s32Width)
camera_height = int(rectAOI.s32Height)
bitspixel = 24  # for color mode: IS_CM_BGR8_PACKED
mem_ptr = ueye.c_mem_p()
mem_id = ueye.int()

# Allocate memory for the image
ueye.is_AllocImageMem(hCam, camera_width, camera_height, bitspixel, mem_ptr, mem_id)
ueye.is_SetImageMem(hCam, mem_ptr, mem_id)

# Start video capture
ueye.is_CaptureVideo(hCam, ueye.IS_WAIT)

# Create a named OpenCV window with adjustable size
cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Video", desired_width, desired_height)

print("Press 'c' to capture predefined sections, 'q' to quit")


def apply_super_resolution(image):
    # Convert the image to RGB|
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Convert the image to a tensor
    image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    # Apply the ESRGAN model
    sr_image_tensor = esrgan_model(image_tensor)
    sr_image = tf.squeeze(sr_image_tensor).numpy()
    sr_image_gray = cv2.cvtColor(sr_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return sr_image_gray


# Function to capture predefined sections
def capture_predefined_sections(frame, sections):
    for i, (x_start, y_start, x_end, y_end) in enumerate(predefined_sections):
        roi_image = frame[y_start + 1 : y_end - 1, x_start + 1 : x_end - 1]
        filename = os.path.join(save_directory, f"section_image_{i}.png")
        cv2.imwrite(filename, roi_image)
        print(f"Captured and saved: {filename}")


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    return equalized


def align_images(master, input_img):
    # Preprocess both images
    master_preprocessed = preprocess_image(master)
    input_preprocessed = preprocess_image(input_img)

    # Use SIFT for feature detection and description
    sift = cv2.SIFT_create(nfeatures=5000)  # Increased number of features
    keypoints1, descriptors1 = sift.detectAndCompute(master_preprocessed, None)
    keypoints2, descriptors2 = sift.detectAndCompute(input_preprocessed, None)

    # Match features using FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

    # Ensure we have enough matches to compute homography
    if len(good_matches) < 4:
        raise ValueError(
            "Not enough matches found to compute homography. Alignment failed."
        )

    # Extract matched points
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    # Compute the homography matrix
    h_matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # Align the input image
    aligned_img = cv2.warpPerspective(
        input_img, h_matrix, (master.shape[1], master.shape[0])
    )
    return aligned_img, h_matrix


def find_defect(img_path):
    master_image_path = "./section 2 clear/New_Master.png"
    input_image_path = img_path

    master_image = cv2.imread(master_image_path)
    input_image = cv2.imread(input_image_path)

    # Resize input image to match the size of the master image
    input_image = cv2.resize(
        input_image, (master_image.shape[1], master_image.shape[0])
    )

    # Align the input image to the master image
    aligned_input_img, h_matrix = align_images(master_image, input_image)

    # Preprocess both images for comparison
    aligned_master_gray = preprocess_image(master_image)
    aligned_input_gray = preprocess_image(aligned_input_img)

    # Compute the absolute difference
    difference = cv2.absdiff(aligned_master_gray, aligned_input_gray)
    difference_canny = cv2.Canny(difference, 100, 200)
    cv2.imshow("Difference Canny", difference_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("difference.png", difference)

    # Threshold the difference image
    _, thresholded_diff = cv2.threshold(difference, 70, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresholded_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("Thresholded Diff.png", thresholded_diff)

    # Find contours of differences
    contours, _ = cv2.findContours(
        thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Highlight differences on the aligned input image
    highlighted_image = aligned_input_img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(highlighted_image, [contour], -1, (0, 0, 255), 2)

    cv2.imshow("Defects", highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Start the live video loop
while True:
    # Create the image buffer for capturing frames
    image_buffer = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)

    # Capture an image frame
    ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)

    # Copy image data to the buffer
    ueye.is_CopyImageMem(hCam, mem_ptr, mem_id, image_buffer.ctypes.data)

    # Resize the captured frame to the desired dimensions
    resized_frame = cv2.resize(image_buffer, (desired_width, desired_height))

    # Draw predefined sections on the frame
    for x_start, y_start, x_end, y_end in predefined_sections:
        cv2.rectangle(resized_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

    # Display the frame with predefined sections
    cv2.imshow("Live Video", resized_frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, capture predefined sections
    if key == ord("c"):
        capture_predefined_sections(resized_frame, predefined_sections)
        print("Sections captured.")

        # Load the captured section image
        section_image_path = os.path.join(save_directory, "section_image_0.png")

        find_defect(section_image_path)

    # If 'q' is pressed, exit the loop
    if key == ord("q"):
        break

# Release the camera and memory
ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
ueye.is_ExitCamera(hCam)

# Destroy the OpenCV window
cv2.destroyAllWindows()
