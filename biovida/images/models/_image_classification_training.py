# coding: utf-8

"""

    Model Training
    ~~~~~~~~~~~~~~

"""
import sys
SAVE_PATH = ""
TRAINING_DATA_PATH = "/biovida_train_val_test"  # path to the training/val/test data
BIOVIDA_PATH = "/BioVida"  # path to the library -- only needed if BioVida is not installed.

sys.path.append(BIOVIDA_PATH)

# General Imports
from keras import backend as K
K.set_image_dim_ordering('th')

from biovida.images.models.image_classification import ImageClassificationCNN


models = ['default', 'squeezenet']


# ------------------------------------------------------------------------------------------
# Image Classification
# ------------------------------------------------------------------------------------------


def select_model():
    print("The following models can be used:\n")
    for e, m in enumerate(models, start=1):
        print("{0}.".format(str(e)), "'{0}'".format(m.replace("_", " ")))

    model_to_use_no = -1
    while model_to_use_no < 1 or model_to_use_no > len(models):
        model_to_use_no = int(input("Please enter the number of "
                                    "the model you wish to use: "))

    model_to_use = models[model_to_use_no - 1]
    print("Selected Model:", model_to_use)
    return model_to_use


def _image_recognition_cnn_training(epochs, model_to_use, save_name):
    """

    Train the model.

    :param epochs: number of epochs
    :type epochs: ``int``
    :param save_name: the name of the weights to be saved
    :type save_name: ``str``
    """
    ircnn = ImageClassificationCNN(TRAINING_DATA_PATH)
    ircnn.convnet(model_to_use=model_to_use)

    ircnn.fit(epochs=epochs)
    ircnn.save(save_name, path=SAVE_PATH)


save_name = input("Please enter 'save name' for the output file: ")
epochs = int(input("Please enter the number of epochs: "))
model_to_use = select_model()

_image_recognition_cnn_training(epochs, model_to_use, save_name)
