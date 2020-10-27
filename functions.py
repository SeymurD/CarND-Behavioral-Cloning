import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.utils import shuffle


# Function to read in the CSV data
def read_data(paths):
    img_paths = []
    angles = []
    for path in paths:
        with open(path, newline='') as f:
            driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

        print("Driving data has %d samples.", len(driving_data))

        # Pull each image (columns 1-3), corresponding angular values, throttle, breaking and vehicle speed
        for row in driving_data[1:]:
            # Remove rows that show car not moving, speed ~= 0
            if float(row[6]) < 0.1:
                continue

            # Adjust steering angles as seen from left/right camera as they should not have
            # vectors that remain parallel to the center camera, but rather converge towards
            # the vector generated from the center camera
            steering_center = float(row[3])
            correction = 0.315
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
def generator(paths, angles, train=False, batchsize=128):
    X = []  # inputs
    y = []  # target
    paths, angles = shuffle(paths, angles)

    while True:
        for i in range(len(angles)):
            image = cv2.imread(paths[i])
            angle = angles[i]
            image = image_preprocess(image)
            # Training modifiers
            if train:
                image, angle = random_distort(image, angle)
                # image = transform_image(image, 40, 0, 10, brightness=1)
                # Flip image with probability of 50%
                # p = np.random.choice(2, 1, p=[0.5, 0.5])
                # if p == 1:
                #     image = np.fliplr(image)
                #     angle = -angle
            # Store in batch variables
            X.append(image)
            y.append(angle)

            if len(X) == batchsize:
                yield np.array(X), np.array(y)
                # Clear batch variables and shuffle
                X, y = ([], [])
                paths, angles = shuffle(paths, angles)

            # flip horizontally and invert steer angle, if magnitude is > 0.33
            if abs(angle) > 0.33:
                image = cv2.flip(image, 1)
                angle *= -1
                X.append(image)
                y.append(angle)
                if len(X) == batchsize:
                    yield np.array(X), np.array(y)
                    X, y = ([], [])
                    paths, angles = shuffle(paths, angles)


# Data pre-processing
def data_preprocess(paths, angles):
    paths, angles = shuffle(paths, angles)
    # Evaluate average frequency
    hist_bins = 81  # 0.1 spacing from -1,1 plus 0
    avg_freq = len(angles) / (hist_bins/2)
    noise = 350    # account for data noise, tune param
    # Visualize histogram
    hist, bins = steer_hist(angles, 0, avg_freq)

    # indices to reduce over-representation of steering angles,
    # reduce multimodal histogram to be approximately uniform
    keep_probs = []
    target = avg_freq * 0.5 # visually inspect the shape and distribution of the steering angles
    for i in range(len(hist)):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1. / (hist[i] / target))
    remove = []
    for i in range(len(angles)):
        if angles[i] > 0.4 or angles[i] < -0.4:
            remove.append(i)
            continue
        for j in range(hist_bins):
            if angles[i] > bins[j] and angles[i] <= bins[j + 1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove.append(i)


    angles = np.delete(angles, remove, axis=0)
    paths = np.delete(paths, remove, axis=0)
    print("Number of over-represented data points:", len(remove))
    print("Total data samples remaining:", len(angles))
    steer_hist(angles, 1, avg_freq)

    return paths, angles

# Visualize steering histogram
def steer_hist(angles, count, avg_freq=0):
    # Visualize distribution of steering angles
    hist, bins = np.histogram(angles, bins=np.linspace(-1, 1, 81))
    mid = (bins[:-1] + bins[1:]) / 2
    width = 0.8 * (bins[1] - bins[0])
    plt.bar(mid, hist, align='edge', width=width)
    plt.xticks(np.linspace(-1, 1, 21), ha='center', rotation=45)
    plt.hlines(avg_freq, -1, 1)
    plt.savefig('./media/hist_dist{0}.png'.format(count))
    plt.show()

    # Print Kolmogorov-Smirnov test metrics
    print(stats.kstest(angles, 'norm'))

    return hist, bins

# Image pre-processing
def image_preprocess(img):
    # Crop the image
    img = img[50:120, :, :]     # 66 pixel vertically is the smallest accepted into Nvidia network
    # Resize based on Nvidia paper
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    # Add Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Converted to YUV colorspace
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

# Image augmentation for robustness of CNN
def transform_image(img, ang_range, shear_range, trans_range, brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    #tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_x = 0
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    Shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,Shear_M,(cols,rows))

    # Brightness
    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img

# Modify brightness of image
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_distort(img, angle):
    '''
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:, :, 0] + value) > 255
    if value <= 0:
        mask = (new_img[:, :, 0] + value) < 0
    new_img[:, :, 0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h, w = new_img.shape[0:2]
    mid = np.random.randint(0, w)
    factor = np.random.uniform(0.6, 0.8)
    if np.random.rand() > .5:
        new_img[:, 0:mid, 0] *= factor
    else:
        new_img[:, mid:w, 0] *= factor
    # randomly shift horizon
    h, w, _ = new_img.shape
    horizon = 2 * h / 5
    v_shift = np.random.randint(-h / 8, h / 8)
    pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
    pts2 = np.float32([[0, horizon + v_shift], [w, horizon + v_shift], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)
