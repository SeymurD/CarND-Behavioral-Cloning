import csv
import numpy as np
import cv2

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
        steering_center = row[3]
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
        cv2.imshow("Sample", img)
        cv2.waitKey(0)


