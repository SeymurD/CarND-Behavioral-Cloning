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
csv_path = './training_data/driving_log.csv'

# Read image data, image resolution is 160x320 (h, w)
img_paths, angles = read_data(csv_path)

# Shuffle the images & angles in unison before visualization
img_paths, angles = shuffle(img_paths, angles)

# Visualize some images
#visualize_data(img_paths, 5)

# Train-Test split
img_path_train, img_path_test, angles_train, angles_test = \
    train_test_split(img_paths, angles, test_size=0.05, random_state=42)
print('Train:', img_path_train.shape, angles_train.shape)
print('Test:', img_path_test.shape, angles_test.shape)

# --- Model Initialization ---
model = Sequential()

net(model)  # adds layers
# ------ Model End Init ------

# Compile and train the model
model.compile(optimizer=Adam(lr=1e-4), loss='mse')

batchsize = 64
# Generators for train, validate and test sets
train_gen = generator(img_path_train, angles_train, batchsize=batchsize)
valid_gen = generator(img_path_train, angles_train, batchsize=batchsize)
test_gen = generator(img_path_test, angles_test, batchsize=batchsize)

# Set checkpoint
checkpoint = ModelCheckpoint(filepath='./model/model.{epoch:02d}-{val_loss:.2f}.h5')

# Train the model
history = model.fit_generator(train_gen, validation_data=valid_gen,
                              validation_steps=len(img_path_test) // batchsize,
                              steps_per_epoch=len(img_path_train) // batchsize,
                              epochs=50, verbose=1, callbacks=[checkpoint])

# Print results
print('Test Loss:', model.evaluate_generator(test_gen, 128))
print(model.summary())

# Save weights
model.save_weights('./model/model')
