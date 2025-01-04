import os

def rename_files(directory, prefix="bad_", extension=".png"):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Sort the files to maintain order
    files.sort()
    
    # Loop through each file and rename it
    for index, filename in enumerate(files):
        # Construct the new filename
        new_filename = f"{prefix}{index + 1}{extension}"
        
        # Get the full path to the old and new filenames
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {old_filepath} to {new_filepath}")

# Define the directory containing the files to rename
directory = "./generated_images/incorrect"

# Call the rename_files function
rename_files(directory)