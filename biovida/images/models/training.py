"""

    Training Models for Image Classification Tasks
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# see: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Imports
import os
import numpy as np
from random import shuffle
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Problem: ValueError: Negative dimension size caused by subtracting 2 from 1
# Solution: replace "th" with "tf" in ~/.keras/keras.json.

# ---------------------------------------------------------------------------------------------
# Getting the Data
# ---------------------------------------------------------------------------------------------

from biovida.images.models.temp import data_path

# ---------------------------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------------------------

# dimensions of our images.
img_width, img_height = 150, 150

# Define data location
train_data_dir = os.path.join(data_path, "train")
validation_data_dir = os.path.join(data_path, "validation")
# nb_train_samples = number_of_images_in_dir(train_data_dir)
# nb_validation_samples = number_of_images_in_dir(validation_data_dir)
nb_epoch = 5

# ------------------------------------------------
# Define the Model
# ------------------------------------------------

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# ------------------------------------------------
# Complile the model
# ------------------------------------------------

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

# This is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# This is the augmentation configuration we will use for testing: only rescaling
test_datagen = ImageDataGenerator(rescale=1.0/255)

# This is a generator that will read pictures found in subfolers of 'data/train', and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=32,
                                                    class_mode='binary')

train_generator.nb_sample

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(directory=validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=32,
                                                        class_mode='binary')


model.fit_generator(generator=train_generator,
                    samples_per_epoch=train_generator.nb_sample,
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=validation_generator.nb_sample)


# Save the weights
# model.save(os.path.join(data_path, "valid_grid_1139.h5"))

# ---------------------------------------------------------------------------------------------
# Feeding in Raw Data
# ---------------------------------------------------------------------------------------------


def img_processing(img_path):
    """

    :param img_path:
    :return:
    """
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)























































































