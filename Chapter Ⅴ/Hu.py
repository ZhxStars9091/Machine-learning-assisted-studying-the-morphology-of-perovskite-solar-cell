import cv2
import numpy as np
import csv
import os

image_folder = 'C:/Users/XXX'
csv_filename = 'C:/Users/XXX.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7"])

    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(binary_image)
            hu_moments = cv2.HuMoments(moments)
            hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
            print(f"Processing {filename}...")
            for i, moment in enumerate(hu_moments_log):
                print(f"Hu[{i + 1}] = {moment[0]:.5e}")
            writer.writerow([filename] + [moment[0] for moment in hu_moments_log])

print(f" {csv_filename}")
