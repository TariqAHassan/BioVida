"""

    Model Training
    ~~~~~~~~~~~~~~

"""
# General Imports
import numpy as np
import scipy.misc
from biovida.images.models.temp import convent_data_path
from keras.preprocessing.image import load_img

# Import the Image Recognition Convnet
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

# _image_rcognition_cnn_training(nb_epoch, training_data_path, save_name)












































