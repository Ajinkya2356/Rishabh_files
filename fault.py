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
    img = img.convert("L")  # Convert to grayscale
    pixels = list(img.getdata())
    color_counts = Counter(pixels)
    background_color = color_counts.most_common(1)[0][0]
    return background_color


# Function to detect segments in an image
def detect_segments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 184, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = [contour for contour in contours if cv2.contourArea(contour) > 10]
    return segments


# Function to create an image with missing segments
def create_image_with_missing_segments(
    image, missing_segments, output_path, background_color
):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_with_segments = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for segment in missing_segments:
        cv2.drawContours(
            image_with_segments,
            [segment],
            -1,
            (background_color, background_color, background_color),
            -1,
        )
    cv2.imwrite(output_path, image_with_segments)


# Main function to process images from the input directory
def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        cropped_image = image[5:h-5, 5:w-5]
        segments = detect_segments(cropped_image)
        if segments:
            num_segments = random.randint(1, len(segments))
            selected_segment = random.sample(segments, num_segments)
            background_color = get_background_color(image_path)

            # Create an image with one random segment missing
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(image_file)[0]}_missing_segment.jpeg"
            )
            create_image_with_missing_segments(
                cropped_image, selected_segment, output_path, background_color
            )


# Example usage
input_directory = "./Analog_mobile/correct"
output_directory = "./Analog_mobile/incorrect"
process_images(input_directory, output_directory)
