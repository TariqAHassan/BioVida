"""

    Model Training
    ~~~~~~~~~~~~~~

    WARNING:
        this script is configured to use
        THEANO as a computational back end.
        To use TensorFlow, make the following
        change:

        K.set_image_dim_ordering('th')

        to

        K.set_image_dim_ordering('tf')

"""
import os
import sys
biovida_path = input("Please enter the full path to BioVida: ")
sys.path.append(biovida_path)
from biovida.images.models._training_data_path import training_data_path

# General Imports
import numpy as np
import scipy.misc
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.preprocessing.image import load_img
from biovida.images.models.img_classification import ImageRecognitionCNN


# ------------------------------------------------------------------------------------------
# Image Classification
# ------------------------------------------------------------------------------------------


def _image_rcognition_cnn_training(nb_epoch, training_data_path, save_name):
    """

    Train the model.

    :param nb_epoch:
    :param training_data_path:
    :param save_name:
    :return:
    """

    ircnn = ImageRecognitionCNN(training_data_path)
    ircnn.convnet()

    ircnn.fit(nb_epoch=nb_epoch)
    ircnn.save(save_name)


save_name = input("Please enter the name of the file: ")
iters = int(input("Please enter the number of iterations: "))
_image_rcognition_cnn_training(iters, training_data_path, save_name)
























