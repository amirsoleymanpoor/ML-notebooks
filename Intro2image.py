#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the first image using PIL
image_path_1 = 'C:/Users/test/Desktop/FB_IMG_1699980572276 (1).jpg'
image = Image.open(image_path_1)
data = np.array(image)

# Display the image
plt.imshow(data)
plt.show()

# Example operations on the image
print(data.shape)
print(data[0, 1000])

data[:, :, 1] = 0
data[:, :, 2] = 0
plt.imshow(data)
plt.show()

# Load the second image from the correct local path
image_path_2 = 'C:/Users/test/Desktop/3333333333.jpg'
image = Image.open(image_path_2)
data = np.array(image)

# Modify the image
data[0:380, 0:350, 0] = 0
data[0:750, 0:750, 1] = 255
data[0:100, 0:100, 2] = 0

# Visualize the result
plt.imshow(data)
plt.show()

# RGB to Grayscale Conversion Function
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(int)

# Example usage of the rgb2gray function
img = Image.open(image_path_2)
Amirsol = rgb2gray(np.array(img))

# Change the image orientation from vertical to horizontal
Amirsol_horizontal = np.transpose(Amirsol)

# Rotate the image 180 degrees
rotated_img = np.rot90(Amirsol, 2)

# Display the horizontally flipped image
plt.imshow(Amirsol_horizontal, cmap='gray')
plt.show()

# Visualization of Amirsol_horizontal in Grayscale
Amirsol_horizontal[1300:1600, 800:1700] = 0
plt.imshow(Amirsol_horizontal, cmap=plt.get_cmap('gray'))
plt.show()
