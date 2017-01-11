"""

    Image Classification
    ~~~~~~~~~~~~~~~~~~~~

"""
# see: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# this also helped: https://blog.rescale.com/neural-networks-using-keras-on-rescale/

# Imports
import os
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Problem: ValueError: Negative dimension size caused by subtracting 2 from 1
# Solution: replace "th" with "tf" in ~/.keras/keras.json.
# Note: `MaxPooling2D` has a `dim_ordering` param which can do the same thing.
# When deployed, dim_ordering should equal 'tf' to avoid users having this problem.
# The same goes to all function below which have this param.

# ---------------------------------------------------------------------------------------------
# Getting the Data
# ---------------------------------------------------------------------------------------------

from biovida.images.models.temp import data_path
from biovida.support_tools.support_tools import n_sub_dirs

# ---------------------------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------------------------

# dimensions of our images.
img_width, img_height = 150, 150

# Define data location
train_data_dir = os.path.join(data_path, "train")
validation_data_dir = os.path.join(data_path, "validation")
# nb_train_samples = number_of_images_in_dir(train_data_dir)
# nb_validation_samples = number_of_images_in_dir(validation_data_dir)

# ---------------------------------------------------------------------------------------------
# Data Stream
# ---------------------------------------------------------------------------------------------

# This is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1.0/255
                                   , shear_range=0.2
                                   , zoom_range=0.2
                                   , horizontal_flip=True)

# This is the augmentation configuration we will use for testing: only rescaling
test_datagen = ImageDataGenerator(rescale=1.0/255)

# This is a generator that will read pictures found in subfolers of 'data/train',
# and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=32,
                                                    class_mode='categorical')

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(directory=validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=32,
                                                        class_mode='categorical')
# train_generator.image_shape
# validation_generator.image_shape

# ---------------------------------------------------------------------------------------------
# Define the Model
# ---------------------------------------------------------------------------------------------

nb_epoch = 2
nb_classes = n_sub_dirs(train_data_dir)


model = Sequential()

# Convolution layers to generate features
# for the fully connected layers below.
model.add(Convolution2D(32, 3, 3,  input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layers, i.e., a standard
# neural net being fed features from above.
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
# Output
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# ------------------------------------------------
# Complile the model
# ------------------------------------------------

model.compile(loss='categorical_crossentropy'
              , optimizer='rmsprop'
              , metrics=['accuracy'])


model.fit_generator(generator=train_generator,
                    samples_per_epoch=train_generator.nb_sample,
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=validation_generator.nb_sample)


# Save the weights
# model.save(os.path.join(data_path, "arrows_boarder_graphs_grids_text_valid_1.h5"))























































