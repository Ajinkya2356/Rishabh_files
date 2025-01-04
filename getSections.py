import os
from PIL import Image
import cv2

def select_roi(image_path, window_name="Select ROI", window_width=400, window_height=500):
    image = cv2.imread(image_path)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    roi = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi

def crop_and_save_images(input_dir, output_dir, num_copies=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_dir, filename)
            roi = select_roi(image_path)
            image = Image.open(image_path)
            
            # Convert ROI from (x, y, w, h) to (left, upper, right, lower)
            left = roi[0]
            upper = roi[1]
            right = roi[0] + roi[2]
            lower = roi[1] + roi[3]
            cropped_image = image.crop((left, upper, right, lower))

            for i in range(num_copies):
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_copy{i+1}{os.path.splitext(filename)[1]}")
                cropped_image.save(output_path)

if __name__ == "__main__":
    input_directory = "./Section2images"
    output_directory = "./Section2images_cropped"
    os.makedirs(output_directory, exist_ok=True)

    crop_and_save_images(input_directory, output_directory)