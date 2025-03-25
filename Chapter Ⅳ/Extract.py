import numpy as np
from skimage import io
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import os
import pandas as pd
import warnings
import openpyxl
def calculate_skeleton_length(skeleton):
    return np.sum(skeleton.astype(int))
def process_images(input_folder, output_folder, xlsx_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    result_df = pd.DataFrame(columns=['ImageName', 'SkeletonLength'])
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = io.imread(image_path, as_gray=True)
            threshold_value = threshold_otsu(image)
            binary_image = image > threshold_value
            skeleton = skeletonize(binary_image)
            skeleton_length = calculate_skeleton_length(skeleton)
            output_path = os.path.join(output_folder, f"skeleton_{filename}")
            with warnings.catch_warnings():
                io.imsave(output_path, (skeleton * 255).astype(np.uint8))
            print(f"Processed: {filename}, Skeleton Length: {skeleton_length}")
            result_df = result_df._append({'ImageName': filename, 'SkeletonLength': skeleton_length}, ignore_index=True)
    result_df.to_excel(xlsx_file, index=False)
    print(f"Results saved to {xlsx_file}")

if __name__ == "__main__":
    input_folder = "C:/Users/"
    output_folder = "C:/Users/"
    xlsx_file = "C:/Users/1.xlsx" 
    process_images(input_folder, output_folder, xlsx_file)
