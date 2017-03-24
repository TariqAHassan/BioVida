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
# General Imports
from keras import backend as K
K.set_image_dim_ordering('th')

from biovida.images.models.image_classification import ImageClassificationCNN


# ------------------------------------------------------------------------------------------
# Image Classification
# ------------------------------------------------------------------------------------------


def _image_recognition_cnn_training(epochs, training_data_path, save_name):
    """

    Train the model.

    :param epochs: number of epochs
    :type epochs: ``int``
    :param training_data_path: path to the training data
    :type training_data_path: ``str``
    :param save_name: the name of the weights to be saved
    :type save_name: ``str``
    """
    ircnn = ImageClassificationCNN(training_data_path)
    ircnn.convnet(model_to_use='alex_net')

    ircnn.fit(epochs=epochs)
    ircnn.save(save_name)


save_name = input("Please enter the name of the file: ")
iters = int(input("Please enter the number of iterations: "))
training_data_path = input("Please enter the path to the training data: ")
_image_recognition_cnn_training(iters, training_data_path, save_name)






















