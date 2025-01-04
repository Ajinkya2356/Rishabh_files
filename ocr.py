import cv2
import easyocr
import numpy as np

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Path to the image
image_path = "./gray.jpeg"

# Load the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to isolate segments
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Detect contours of segments
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image (optional)
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# Perform OCR using EasyOCR
results = reader.readtext(gray)

# Draw bounding boxes and recognized text on the image
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    
    # Draw the bounding box
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Draw the text
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Check if all segments are on
expected_segments = 7  # Adjust this based on the number of expected segments
detected_segments = len(contours)

if detected_segments == expected_segments:
    print("All segments are on.")
else:
    print(f"Detected {detected_segments} segments. Some segments are missing or extra.")

# Show the processed image
cv2.imshow("Image with Contours", image_contours)
cv2.imshow("OCR Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()