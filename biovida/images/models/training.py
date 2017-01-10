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

from biovida.support_tools.support_tools import dict_reverse
from biovida.support_tools.support_tools import images_in_dir

# Problem: ValueError: Negative dimension size caused by subtracting 2 from 1
# Solution: replace "th" with "tf" in ~/.keras/keras.json.

# ---------------------------------------------------------------------------------------------
# Getting the Data
# ---------------------------------------------------------------------------------------------

from biovida.images.models.temp import data_path

# ---------------------------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------------------------

# Define data location
train_data_dir = os.path.join(data_path, "train")
validation_data_dir = os.path.join(data_path, "validation")

class ImageRecognitionCNN(object):
    """

    Wraps the Convolutional Neural Network Model used to identify invalid images types.

    """

    def __init__(self, img_size=(150, 150), rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True):
        """

        :param rescale:
        :param shear_range:
        :param zoom_range:
        :param horizontal_flip:
        """
        # Params
        self.img_size = img_size
        self.rescale = rescale
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip

        # Data Generators
        self._train_img_datagen = None
        self._test_img_datagen = None

        # Generators
        self._train_generator
        self._validation_generator

        # Model
        self.model = None

    def _model_def_and_compilation(self):
        """

        :return:
        """
        # Define a seqential model
        self.model = Sequential()
        self.model.add(Convolution2D(32, 3, 3, input_shape=(3, self.img_size[0], self.img_size[1])))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        # Compilation
        self.model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    def _image_data_gen_train(self):
        """

        :return:
        """
        self._train_img_datagen = ImageDataGenerator(rescale=self.rescale,
                                                     shear_range=self.shear_range,
                                                     zoom_range=self.zoom_range,
                                                     horizontal_flip=self.horizontal_flip)

    def _image_data_gen_test(self):
        """

        :return:
        """
        self._test_img_datagen = ImageDataGenerator(rescale=self.rescale)

    def generator(self, train_data_dir, validation_data_dir):
        """

        :param train_data_dir:
        :param validation_data_dir:
        :param t_size:
        """
        # Compile the model
        self._model_def_and_compilation()

        # This is a generator that will read pictures found in subfolers of 'data/train',
        # and indefinitely generate batches of augmented image data
        self._train_generator = self._train_img_datagen.flow_from_directory(directory=train_data_dir,
                                                                            target_size=self.img_size,
                                                                            batch_size=32,
                                                                            class_mode='binary')

        # This is a similar generator, for validation data
        self._validation_generator = self._test_img_datagen.flow_from_directory(directory=validation_data_dir,
                                                                                target_size=self.img_size,
                                                                                batch_size=32,
                                                                                class_mode='binary')

    def fit(self, save_path=None, nb_epoch=10):
        """

        :param save_path: ToDo: automatically pick this
        :param nb_epoch:
        :return:
        """
        if any(x is None for x in (self._train_generator, self._validation_generator)):
            raise AttributeError("Generators are not defined. Please call `ImageRecognitionCNN().generator()")

        self.model.fit_generator(generator=self._train_generator,
                                 samples_per_epoch=self._train_generator.nb_sample,
                                 nb_epoch=nb_epoch,
                                 validation_data=self._validation_generator,
                                 nb_val_samples=self._validation_generator.nb_sample)

        if save_path is not None:
            self.model.save(save_path, overwrite=False)

    def load(self, weights_path):
        """

        :param weights_path:
        :return:
        """
        self.model.load(weights_path, overwrite=False)



# Save the weights
# model.save(os.path.join(data_path, "valid_grid_1386.h5"))

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





























































