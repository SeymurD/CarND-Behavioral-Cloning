import cv2
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from functions import *

# Generate file path to training data from CSV file
csv_path = './training_data/driving_log.csv'

# Read image data, image resolution is 160x320 (h, w)
img_paths, angles = read_data(csv_path)

# Shuffle the images & angles in unison before visualization
img_paths, angles = shuffle(img_paths, angles)

# Visualize some images
visualize_data(img_paths, 5)
