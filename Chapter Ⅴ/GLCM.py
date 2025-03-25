import os
import numpy as np
import cv2
import math
import csv
gray_level = 16
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1

def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    if d_x >= 0 or d_y >= 0:
        for j in range(height - d_y):
            for i in range(width - d_x):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0
    else:
        for j in range(height):
            for i in range(width):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0
    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret

def feature_computer(p):
    mean = 0.0
    std = 0.0
    contrast = 0.0
    dissimilarity = 0.0
    homogeneity = 0.0
    asm = 0.0
    energy = 0.0
    entropy = 0.0
    max_probability = 0.0
    gray_levels = len(p)

    for i in range(gray_levels):
        for j in range(gray_levels):
            mean += i * p[i][j]
            max_probability = max(max_probability, p[i][j])

    for i in range(gray_levels):
        for j in range(gray_levels):
            std += ((i - mean) ** 2) * p[i][j]

    std = np.sqrt(std)

    for i in range(gray_levels):
        for j in range(gray_levels):
            contrast += (i - j) ** 2 * p[i][j]
            dissimilarity += abs(i - j) * p[i][j]
            homogeneity += p[i][j] / (1 + (i - j)**2)
            asm += p[i][j] ** 2
            energy += p[i][j] ** 0.5
            if p[i][j] > 0.0:
                entropy += -p[i][j] * math.log(p[i][j])

    return [mean, std, contrast, dissimilarity, homogeneity, asm, energy, entropy, max_probability]

def process_folder(input_folder, output_csv):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if os.path.exists(output_csv):
        os.remove(output_csv)
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)

        try:
            img_shape = img.shape
        except:
            print('imread error:', image_file)
            continue

        img = cv2.resize(img, (img_shape[1] // 2, img_shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        glcm_0 = getGlcm(img_gray, 1, 0)

        features = feature_computer(glcm_0)

if __name__ == '__main__':
    input_folder = "C:/Users/XXX"
    output_csv = "C:/Users/XXX.csv"
    process_folder(input_folder, output_csv)
