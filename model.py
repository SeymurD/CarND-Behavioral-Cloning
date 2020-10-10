import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from functions import *
from architecture import net

# Generate file path to training data from CSV file
csv_path = ['./training_data/driving_log.csv', './training_data_reverse/driving_log.csv']

# Read image data, image resolution is 160x320 (h, w)
img_paths, angles = read_data(csv_path)

# Shuffle the images & angles in unison before visualization
img_paths, angles = shuffle(img_paths, angles)

# Visualize some images
visualize_data(img_paths, 5)

# Process some of the images
img_paths, angles = data_preprocess(img_paths, angles)

# Toggle boolean to control if training is desired or just monitoring initial data
isTrain = True
if isTrain:
    # Train-Test split
    img_path_train, img_path_test, angles_train, angles_test = \
        train_test_split(img_paths, angles, test_size=0.05, random_state=42)
    print('Train:', img_path_train.shape, angles_train.shape)
    print('Test:', img_path_test.shape, angles_test.shape)

    # --- Model Initialization ---
    model = Sequential()

    net(model)  # adds layers, see architecture.py
    # ------ Model End Init ------

    # Compile and train the model
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    batchsize = 128
    # Generators for train, validate and test sets
    train_gen = generator(img_path_train, angles_train, train=True, batchsize=batchsize)
    valid_gen = generator(img_path_train, angles_train, train=False, batchsize=batchsize)
    test_gen = generator(img_path_test, angles_test, train=False, batchsize=batchsize)

    # Set checkpoint
    checkpoint = ModelCheckpoint(filepath='./model/model.{epoch:02d}-{val_loss:.2f}.h5')

    # Train the model
    history = model.fit_generator(train_gen, validation_data=valid_gen,
                                  validation_steps=len(img_path_test) // batchsize,
                                  steps_per_epoch=len(img_path_train) // batchsize,
                                  epochs=100, verbose=1, callbacks=[checkpoint])

    # Print results
    print('Test Loss:', model.evaluate_generator(test_gen, 128))
    print(model.summary())

    # Save weights
    model.save_weights('./model/model.h5')
