"""

    Model Training
    ~~~~~~~~~~~~~~

    WARNING:

    This script is configured to use
    THEANO as a computational back end.
    To use TensorFlow, make the following
    change:

    K.set_image_dim_ordering('th') --> TO --> K.set_image_dim_ordering('tf')

"""
import os
import sys

# Get path to the BioVida library
biovida_path = input("Please enter the full path to BioVida: ")
if not os.path.isdir(biovida_path):
    raise NotADirectoryError("'{0}' is an invalid path.".format(biovida_path))
sys.path.append(biovida_path)

# Get path to the training data
try:
    from biovida.images.models._training_data_path import training_data_path
except:
    training_data_path = input("Please enter the full path to the training data: ")
    if not os.path.isdir(biovida_path):
        raise NotADirectoryError("'{0}' is an invalid path.".format(training_data_path))

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

    :param nb_epoch: number of epochs
    :type nb_epoch: ``int``
    :param training_data_path: path to the training data
    :type training_data_path: ``str``
    :param save_name: the name of the weights to be saved
    :type save_name: ``str``
    """
    ircnn = ImageRecognitionCNN(training_data_path)
    ircnn.convnet(model_to_use='vgg19')

    ircnn.fit(nb_epoch=nb_epoch)
    ircnn.save(save_name)


save_name = input("Please enter the name of the file: ")
iters = int(input("Please enter the number of iterations: "))
_image_rcognition_cnn_training(iters, training_data_path, save_name)
























