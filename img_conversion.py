from PIL import Image
from pathlib import Path
import os

def jpeg_to_jpg(img_folder):
    img_folder = Path(img_folder)
    # Create a list of all .jpeg files in the specified folder
    img_file_list = img_folder.glob("*.jpeg")
    
    for img_file in img_file_list:


        # Open each .jpeg file
        with Image.open(img_file) as img:

            # Create the output file name by replacing .jpeg with .jpg
            output_file = img_file.with_suffix(".jpg")

            # Save the image as .jpg in the same folder
            img.save(output_file, 'JPEG')

            print('going on')

        os.remove(img_file)
    
    print("Conversion complete!")

def jpeg_to_png(img_folder):
    img_folder = Path(img_folder)
    # Create a list of all .jpeg files in the specified folder
    img_file_list = img_folder.glob("*.jpeg")
    
    for img_file in img_file_list:


        # Open each .jpeg file
        with Image.open(img_file) as img:

            # Create the output file name by replacing .jpeg with .jpg
            output_file = img_file.with_suffix(".png")

            # Save the image as .jpg in the same folder
            img.save(output_file, 'JPEG')

            print('going on')

        os.remove(img_file)
    
    print("Conversion complete!")

