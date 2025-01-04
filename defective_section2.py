import os
import cv2
import numpy as np
import random
from PIL import Image
from collections import Counter
import itertools

# Function to get the background color of a grayscale image
def get_background_color(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    pixels = list(img.getdata())
    color_counts = Counter(pixels)
    background_color = color_counts.most_common(1)[0][0]
    return background_color

# Function to detect segments in an image
def detect_segments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 174, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = [contour for contour in contours if cv2.contourArea(contour) > 10]
    return segments

# Function to apply random brightness and contrast adjustments
def adjust_brightness_contrast(image):
    # Randomly adjust brightness within controlled limits
    brightness = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Randomly adjust contrast within controlled limits
    contrast = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    
    return image

# Function to create an image with missing segments and varying brightness/contrast
def create_image_with_missing_segments(image, missing_segments, output_path, background_color):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_with_segments = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for segment in missing_segments:
        cv2.drawContours(image_with_segments, [segment], -1, (background_color, background_color, background_color), -1)
    
    # Apply random brightness and contrast adjustments
    image_with_segments = adjust_brightness_contrast(image_with_segments)
    
    # Save the image with missing segments and adjustments
    cv2.imwrite(output_path, image_with_segments)

# Main function to generate images with different combinations of missing segments
def generate_images_with_combinations(input_image_path, output_dir, num_images=1500):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(input_image_path)
    segments = detect_segments(image)
    background_color = get_background_color(input_image_path)

    # Generate combinations of missing segments
    segment_combinations = list(itertools.combinations(segments, r) for r in range(1, len(segments) + 1))
    segment_combinations = list(itertools.chain.from_iterable(segment_combinations))
    segment_combinations = random.sample(segment_combinations, min(num_images, len(segment_combinations)))

    # Create images with missing segments
    for i, combination in enumerate(segment_combinations):
        output_path = os.path.join(output_dir, f"bad_{i+1}.png")
        create_image_with_missing_segments(image, combination, output_path, background_color)

# Example usage
input_image_path = "./clear.jpeg"
output_directory = "./Section 2 New/train/class 0"
generate_images_with_combinations(input_image_path, output_directory)