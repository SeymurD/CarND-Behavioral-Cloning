import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.utils import shuffle


# Function to read in the CSV data
def read_data(path):
    with open(path, newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    print("Driving data has %d samples.", len(driving_data))

    img_paths = []
    angles = []
    # Pull each image (columns 1-3), corresponding angular values, throttle, breaking and vehicle speed
    for row in driving_data[1:]:
        # Remove rows that show car not moving, speed ~= 0
        if float(row[6]) < 0.1:
            continue

        # Adjust steering angles as seen from left/right camera as they should not have
        # vectors that remain parallel to the center camera, but rather converge towards
        # the vector generated from the center camera
        steering_center = float(row[3])
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # Center image + angle
        img_paths.append(row[0])
        angles.append(steering_center)
        # Left image + angle
        img_paths.append(row[1])
        angles.append(steering_left)
        # Right image + angle
        img_paths.append(row[2])
        angles.append(steering_right)

    # Convert to numpy arrays
    img_paths = np.array(img_paths)
    angles = np.array(angles)

    return img_paths, angles


# Visualize some of the sample data
def visualize_data(path, batchsize=10):
    for i in range(batchsize):
        img = cv2.imread(path[i])
        img = image_preprocess(img)
        cv2.imshow("Sample", img)
        cv2.waitKey(0)


# Generate batched data to be passed into fit_generator. Training, validation and test data can
# all be generated from the following function
def generator(paths, angles, batchsize=128):
    X = []  # inputs
    y = []  # target
    paths, angles = shuffle(paths, angles)

    while True:
        for i in range(len(angles)):
            image = cv2.imread(paths[i])
            angle = angles[i]
            image = image_preprocess(image)
            # Store in batch variables
            X.append(image)
            y.append(angle)

            if len(X) == batchsize:
                yield np.array(X), np.array(y)
                # Clear batch variables and shuffle
                X, y = ([], [])
                paths, angles = shuffle(paths, angles)


# Data pre-processing
def data_preprocess(paths, angles):
    # Evaluate average frequency
    hist_bins = 41  # 0.1 spacing from -1,1 plus 0
    avg_freq = len(angles) / hist_bins
    noise = 400     # account for data noise, tune param
    # Visualize histogram
    hist, bins = steer_hist(angles, avg_freq)

    # indices to reduce over-representation of steering angles,
    # reduce multimodal histogram to be approximately uniform
    remove = []
    for j in range(len(hist)):
        if hist[j] > (avg_freq + noise):
            for i in range(len(angles)):
                if (angles[i] > (bins[j])) and (angles[i] <= (bins[j+2])):
                    # Drop the values based on 60% chance, value can be tuned later
                    p = np.random.choice(2, 1, p=[0.4, 0.6])
                    if p == 1:
                        remove.append(i)

    angles = np.delete(angles, remove, axis=0)
    paths = np.delete(paths, remove, axis=0)
    print("Number of over-represented data points:", len(remove))
    print("Total data samples remaining:", len(angles))
    steer_hist(angles, avg_freq)

    return paths, angles

# Visualize steering histogram
def steer_hist(angles, avg_freq=0):
    # Visualize distribution of steering angles
    hist, bins = np.histogram(angles, bins=np.linspace(-1, 1, 41))
    mid = (bins[:-1] + bins[1:]) / 2
    width = 0.8 * (bins[1] - bins[0])
    plt.bar(mid, hist, align='edge', width=width)
    plt.xticks(np.linspace(-1, 1, 21), ha='center', rotation=45)
    plt.hlines(avg_freq, -1, 1)
    plt.show()

    # Print Kolmogorov-Smirnov test metrics
    print(stats.kstest(angles, 'norm'))

    return hist, bins

# Image pre-processing and augmentation for robustness of CNN
def image_preprocess(img):
    # Crop the image
    img = img[54:120, :, :]     # 66 pixel vertically is the smallest accepted into Nvidia network
    # Add Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img
