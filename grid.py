from pyueye import ueye
import numpy as np
import cv2
import os
import json

# Define the desired width and height for the live video preview
desired_width = 470  # Set the desired width here
desired_height = 420  # Set the desired height here

# Define the directory to save captured images
save_directory = "./section 2 clear"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define the file to save rectangles
rectangles_file = "rectangles.json"

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

print("Press 'c' to capture images inside the rectangles, 's' to save the rectangles, 'e' to erase the rectangles, 'q' to quit")

# Variables to store rectangles
rectangles = []
drawing = False
start_point = None

# Mouse callback function to draw rectangles
def draw_rectangle(event, x, y, flags, param):
    global rectangles, drawing, start_point
    frame = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 1)
            cv2.imshow("Live Video", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        rectangles.append((start_point, end_point))

# Function to capture images inside the rectangles
def capture_images(frame, rectangles):
    for i, (start_point, end_point) in enumerate(rectangles):
        x_start, y_start = start_point
        x_end, y_end = end_point
        roi_image = frame[y_start:y_end, x_start:x_end]
        filename = os.path.join(save_directory, f"rectangle_image_{i}.png")
        cv2.imwrite(filename, roi_image)
        print(f"Captured and saved: {filename}")

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

    # Draw rectangles on the frame
    for (start_point, end_point) in rectangles:
        cv2.rectangle(resized_frame, start_point, end_point, (0, 255, 0), 1)

    # Display the frame with rectangles
    cv2.imshow("Live Video", resized_frame)

    # Set mouse callback function with the current frame
    cv2.setMouseCallback("Live Video", draw_rectangle, [resized_frame])

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, capture images inside the rectangles
    if key == ord('c'):
        capture_images(resized_frame, rectangles)
        print("Images captured.")

    # If 's' is pressed, save the rectangles to file
    if key == ord('s'):
        with open(rectangles_file, "w") as f:
            json.dump(rectangles, f)
        print("Rectangles saved.")

    # If 'e' is pressed, erase the rectangles
    if key == ord('e'):
        rectangles = []
        print("Rectangles erased.")

    # If 'q' is pressed, exit the loop
    if key == ord('q'):
        break

# Release the camera and memory
ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
ueye.is_ExitCamera(hCam)

# Destroy the OpenCV window
cv2.destroyAllWindows()