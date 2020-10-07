import csv
import numpy as np


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
        if row[6] < 0.1:
            continue
        # Center image + angle
        img_paths.append(row[0])
        angles.append(row[3])
        # Left image + angle
        img_paths.append(row[1])
        angles.append(row[3])
        # Right image + angle
        img_paths.append(row[2])
        angles.append(row[3])

    # Convert to numpy arrays
    img_paths = np.array(img_paths)
    angles = np.array(angles)

    return img_paths, angles

