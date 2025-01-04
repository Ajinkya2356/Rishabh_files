from pyueye import ueye
import numpy as np
import cv2
import os
import random

# Define the desired width and height for the live video preview
desired_width = 500  # Set the desired width here
desired_height = 400  # Set the desired height here

# Define the directory to save captured images
save_directory = "./generate_images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

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

# Set color mode to MONO8 for monochrome camera
ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8)

# Set the area of interest (AOI) to the maximum resolution supported by the camera
max_width = int(sInfo.nMaxWidth)
max_height = int(sInfo.nMaxHeight)
rectAOI.s32X = ueye.int(0)
rectAOI.s32Y = ueye.int(0)
rectAOI.s32Width = ueye.int(max_width)
rectAOI.s32Height = ueye.int(max_height)
ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rectAOI, ueye.sizeof(rectAOI))

# Convert c_int to Python int for actual width and height from the camera
camera_width = int(rectAOI.s32Width)
camera_height = int(rectAOI.s32Height)
bitspixel = 8  # for monochrome mode: IS_CM_MONO8
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

print(
    "Press 'r' to select ROI, 'c' to start capturing, 'd' to pause, 's' to resume, 'q' to quit"
)

# Initialize the ROI variables
roi_selected = False
roi_x, roi_y, roi_width, roi_height = 0, 0, 0, 0

# State variable to control capturing
capturing = False

# Function to apply random augmentations to an image
def augment_image(image):
    # Randomly adjust brightness within controlled limits
    brightness = random.uniform(0.8, 1)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    # Randomly adjust contrast within controlled limits
    contrast = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    return image

# Start the live video loop
while True:
    # Create the image buffer for capturing frames
    image_buffer = np.zeros((camera_height, camera_width), dtype=np.uint8)

    # Capture an image frame
    ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)

    # Copy image data to the buffer
    ueye.is_CopyImageMem(hCam, mem_ptr, mem_id, image_buffer.ctypes.data)

    # Resize the captured frame to the desired dimensions for display
    resized_frame = cv2.resize(image_buffer, (desired_width, desired_height))

    # If ROI has been selected, display a border around it
    if roi_selected:
        # Draw a border around the selected ROI
        cv2.rectangle(
            resized_frame,
            (roi_x, roi_y),
            (roi_x + roi_width, roi_y + roi_height),
            (0, 0, 0),
            2,
        )

    # Display the frame with the ROI border in the OpenCV window
    cv2.imshow("Live Video", resized_frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # Select ROI dynamically when 'r' is pressed
    if key == ord("r"):
        # Use OpenCV's selectROI to allow the user to select the ROI with the mouse
        roi = cv2.selectROI(
            "Live Video", resized_frame, showCrosshair=True, fromCenter=False
        )

        # Extract the ROI coordinates
        roi_x, roi_y, roi_width, roi_height = (
            int(roi[0]),
            int(roi[1]),
            int(roi[2]),
            int(roi[3]),
        )

        # Mark ROI as selected
        roi_selected = True

    # Start capturing frames when 'c' is pressed
    elif key == ord("c"):
        capturing = True

    # Pause capturing frames when 'd' is pressed
    elif key == ord("d"):
        capturing = False

    # Resume capturing frames when 's' is pressed
    elif key == ord("s"):
        capturing = True

    # Capture and save frames if capturing is enabled and ROI is selected
    if capturing and roi_selected:
        # Extract the ROI from the current resized frame without the border
        roi_image = resized_frame[
            roi_y : roi_y + roi_height, roi_x : roi_x + roi_width
        ].copy()

        """ augmented_image = augment_image(roi_image) """
        augmented_image = roi_image
        # Generate a unique filename for the captured image
        filename = os.path.join(
            save_directory, f"image_{len(os.listdir(save_directory)) + 1}.png"
        )

        # Save the augmented image to the specified directory
        cv2.imwrite(filename, augmented_image)
        print(f"Captured and saved: {filename}")

    # Quit the live video preview when 'q' is pressed
    if key == ord("q"):
        break

# Release the camera and memory
ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
ueye.is_ExitCamera(hCam)

# Destroy the OpenCV window
cv2.destroyAllWindows()