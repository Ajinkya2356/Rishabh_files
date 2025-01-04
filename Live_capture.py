from pyueye import ueye
import numpy as np
import cv2
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import easyocr

reader = easyocr.Reader(["en"])

# Define the desired width and height for the live video preview
desired_width = 470  # Set the desired width here
desired_height = 420  # Set the desired height here

# Define the directory to save captured images
save_directory = "./section 2 clear"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define the predefined sections (rectangles) to capture
# Each tuple represents (x_start, y_start, x_end, y_end)
predefined_sections = [
    (10, 39, 111, 355),
    (110, 1, 433, 30),
    (355, 30, 448, 415),
    (111, 30, 355, 355),
    (11, 355, 355, 415),
]

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

# Path to your saved model architecture and weights
MODEL_PATH = "./Custom Model/analog_pretrain.h5"

model = load_model(MODEL_PATH)

# Define image size expected by the model
IMG_HEIGHT, IMG_WIDTH = 224, 224


# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the image to match the model's input shape.
    Args:
        image: PIL.Image object
    Returns:
        Preprocessed image ready for prediction
    """
    # Convert the image to grayscale (if required for your model)
    image = image.convert("RGB")  # 'L' mode for grayscale
    # Resize the image
    image = image.resize(
        (
            IMG_HEIGHT,
            IMG_WIDTH,
        )
    )
    # Convert to numpy array
    img_array = img_to_array(image) / 255.0  # Normalize
    # Add batch dimension and channel dimension
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array


# Function to make predictions
def predict_image(image):
    """
    Predict the class of the image using the loaded model.
    Args:
        image: Preprocessed image array
    Returns:
        Prediction result (Defective or Non-Defective)
    """
    prediction = np.round(model.predict(image)[0][0]).astype(int)
    print(model.predict(image))
    print(prediction)
    return "Class 1" if prediction == 1 else "Class 0"


# Function to capture predefined sections
def capture_predefined_sections(frame, sections):
    for i, (x_start, y_start, x_end, y_end) in enumerate(predefined_sections):
        roi_image = frame[y_start + 1 : y_end + 1, x_start + 1 : x_end + 1]
        filename = os.path.join(save_directory, f"section_image_{i}.png")
        cv2.imwrite(filename, roi_image)
        print(f"Captured and saved: {filename}")


def ocr(ocr_images):
    for image_path in ocr_images:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        labels = []
        for bbox, text, prob in results:
            if prob < 0.5:
                continue
            labels.append(text)
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                image,
                text,
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        cv2.imshow("OCR Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("OCR Results : ", labels)


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
        ocr_image_path_1 = os.path.join(save_directory, "section_image_1.png")
        ocr_image_path_2 = os.path.join(save_directory, "section_image_2.png")
        ocr_images = [
            ocr_image_path_1,
            ocr_image_path_2,
        ]
        section_image = Image.open(section_image_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(section_image)

        # Predict the result
        prediction = predict_image(preprocessed_image)

        # Print the prediction result
        print(f"Prediction for section_image_0.png: {prediction}")

        ocr_result = ocr(ocr_images)

    # If 'q' is pressed, exit the loop
    if key == ord("q"):
        break

# Release the camera and memory
ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
ueye.is_ExitCamera(hCam)

# Destroy the OpenCV window
cv2.destroyAllWindows()
