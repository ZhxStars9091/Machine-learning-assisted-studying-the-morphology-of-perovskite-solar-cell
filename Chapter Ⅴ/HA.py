import os
import numpy as np
from PIL import Image
import csv
from scipy.stats import skew, kurtosis

def calculate_gray_features(image):
    gray_image = image.convert('L')

    gray_array = np.array(gray_image)

    mean = np.mean(gray_array)
    variance = np.var(gray_array)

    energy = np.sum(gray_array ** 2)

    contrast = np.mean((gray_array - mean) ** 2)

    hist, _ = np.histogram(gray_array, bins=256, range=[0, 255])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    skewness = skew(gray_array.flatten())
    kurt = kurtosis(gray_array.flatten())

    return [variance, energy, contrast, entropy, skewness, kurt]

folder_path = 'C:/Users/XXX'

csv_file = 'C:/Users/XXX.csv'

features_list = []

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        features = calculate_gray_features(image)
        features_list.append([filename] + features)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Variance', 'Energy', 'Contrast', 'Entropy', 'Skewness', 'Kurtosis'])
    writer.writerows(features_list)

print("Features saved to", csv_file)
