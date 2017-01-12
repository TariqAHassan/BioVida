"""

    Model Training
    ~~~~~~~~~~~~~~

"""
# General Imports
from biovida.images.models.temp import data_path

# ------------------------------------------------------------------------------------------
# Image Classification
# ------------------------------------------------------------------------------------------

# Import the CNN Model
from biovida.images.models.img_classification import ImageRecognitionCNN

# Define Class and Run
ircnn = ImageRecognitionCNN(data_path)
ircnn.conv_net()

ircnn.fit(nb_epoch=10)
ircnn.save("10_epoch_new_model_2")
















































































