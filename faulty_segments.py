import cv2
import numpy as np
import itertools
import os
import random
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# Load the image
image_path = "./gray.jpeg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# List to store the coordinates of the segments
segments = []

# Filter and draw the detected segments
for contour in contours:
    # Filter out small contours that are not segments
    if cv2.contourArea(contour) > 10:  # Adjust the threshold as needed
        segments.append(contour)
        # Draw the contour on the image (optional)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Display the image with detected segments (optional)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Segments")
plt.axis('off')
plt.show()

# Generate random sample of segments
selected_segments = random.sample(segments, min(20, len(segments)))

# Generate all possible combinations of missing segments
def generate_combinations(segments):
    all_combinations = []
    for r in range(1, len(segments) + 1):
        combinations = list(itertools.combinations(segments, r))
        all_combinations.extend(combinations)
        if len(all_combinations) >= 10000:
            break
    return all_combinations

segment_combinations = generate_combinations(selected_segments)

# Limit the number of combinations to 5000
segment_combinations = random.sample(segment_combinations, min(1, len(segment_combinations)))

# Function to get the background color of the image
def get_background_color(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = list(img.getdata())
    color_counts = Counter(pixels)
    background_color = color_counts.most_common(1)[0][0]
    return background_color

background_color = get_background_color(image_path)
print("Background color:", background_color)

# Create output directory
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)

# Function to create an image with missing segments
def create_image_with_missing_segments(image, missing_segments, output_path):
    image = image.copy()
    for segment in missing_segments:
        cv2.drawContours(image, [segment], -1, background_color, -1)  # Fill the segment with background color
    cv2.imwrite(output_path, image)

# Create images with missing segments
for i, combination in enumerate(segment_combinations):
    output_path = os.path.join(output_dir, f"image_{i+1}.jpeg")
    create_image_with_missing_segments(gray, combination, output_path)