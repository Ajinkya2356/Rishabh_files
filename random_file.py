import os
import random
import shutil


def move_random_images(input_dir, output_dir, num_files):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all image files in the input directory
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    # Check if the number of files to move is greater than the available files
    if num_files > len(image_files):
        raise ValueError(
            "Number of files to move is greater than the number of available files"
        )

    # Randomly select the specified number of files
    selected_files = random.sample(image_files, num_files)

    # Move the selected files to the output directory
    for file in selected_files:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(output_dir, file)
        shutil.move(src_path, dest_path)
        print(f"Moved {file} to {output_dir}")


if __name__ == "__main__":
    input_directory = "./Clear Section 2/train/class 0"
    output_directory = "./Clear Section 2/test/class 0"
    number_of_files_to_move = 200

    move_random_images(input_directory, output_directory, number_of_files_to_move)
