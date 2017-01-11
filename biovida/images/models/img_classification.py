"""

    Image Classification
    ~~~~~~~~~~~~~~~~~~~~

"""
# see: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# this also helped: https://blog.rescale.com/neural-networks-using-keras-on-rescale/

# Imports
import os
import numpy as np
# from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Problem: ValueError: Negative dimension size caused by subtracting 2 from 1
# Solution: replace "tf" with "th" in ~/.keras/keras.json.
# Note: `MaxPooling2D` has a `dim_ordering` param which can do the same thing.
# When deployed, `dim_ordering` should equal 'th' to avoid users having this problem.
# The same goes to all function below which have this param.

# from biovida.images.models.temp import data_path


class ImageRecognitionCNN(object):
    """

    Convolutional Neural Network used to Model and Detect the Presence of Invalid Images.
    Keras Sequential Model used.

    :param data_path: path to the directory with the subdirectories entitled 'train' and 'validation'.
                      This directory *must* have this structure. Defaults to ``None`` (to be use when loaded
                      pre-computed weights).
    :type data_path: ``str``
    :param img_shape: the (width, height) to rescale the images to. Elements must be ``ints``. Defaults to (150, 150).
    :type img_shape: ``tuple`` or ``list``.
    :param rescale: See: ``keras.preprocessing.image.ImageDataGenerator()``.
    :type rescale: ``float``
    :param shear_range: See: ``keras.preprocessing.image.ImageDataGenerator()``.
    :type shear_range: ``float``
    :param zoom_range: See: ``keras.preprocessing.image.ImageDataGenerator()``.
    :type zoom_range: ``float``
    :param horizontal_flip: See: ``keras.preprocessing.image.ImageDataGenerator()``.
    :type horizontal_flip: ``bool``
    :param batch_size: Samples to propagate through the model.
                       See: ``keras.preprocessing.ImageDataGenerator().flow_from_directory()``.
                       Defaults to 32.
    :type batch_size: ``int``
    :param dim_ordering: one of: 'tf', 'th'. If keras raises an error of the form:
                                  "'ValueError' a Negative dimension size caused by..." consider
                                  changing the parameter to `th'. Defaults to 'tf'.
                                  See: ``keras.layers.Convolution2D()`` ; ``keras.layers.MaxPooling2D()``.
    :type dim_ordering: ``str``
    """

    def __init__(self
                 , data_path=None
                 , img_shape=(150, 150)
                 , rescale=1.0/255
                 , shear_range=0.2
                 , zoom_range=0.2
                 , horizontal_flip=True
                 , batch_size=32
                 , dim_ordering='tf'):

        self._data_path = data_path
        self._img_shape = img_shape
        self._rescale = rescale
        self._shear_range = shear_range
        self._zoom_range = zoom_range
        self._horizontal_flip = horizontal_flip
        self._batch_size = batch_size
        self._dim_ordering = dim_ordering

        # Define data location
        self._train_data_dir = os.path.join(self._data_path, "train")
        self._validation_data_dir = os.path.join(self._data_path, "validation")

        # Data Streams
        self._train_generator = None
        self._validation_generator = None
        self._data_classes = None

        # The model itself
        self.model = None
        self._nb_classes_default = 6

    def _train_gen(self):
        """

        Use of ``keras.preprocessing.image.ImageDataGenerator()`` to generate training stream.

        """
        # Train augmentation configuration
        train_datagen = ImageDataGenerator(rescale=self._rescale
                                           , shear_range=self._shear_range
                                           , zoom_range=self._zoom_range
                                           , horizontal_flip=self._horizontal_flip)

        # Indefinitely generate batches of augmented train image data
        self._train_generator = train_datagen.flow_from_directory(directory=self._train_data_dir,
                                                                  target_size=self._img_shape,
                                                                  class_mode='categorical',
                                                                  batch_size=self._batch_size)

    def _val_gen(self):
        """

        Use of ``keras.preprocessing.image.ImageDataGenerator()`` to generate validation stream.

        """
        # Test augmentation configuration
        validation_datagen = ImageDataGenerator(rescale=self._rescale)

        # This is a similar generator, for validation data
        self._validation_generator = validation_datagen.flow_from_directory(directory=self._validation_data_dir,
                                                                            target_size=self._img_shape,
                                                                            class_mode='categorical',
                                                                            batch_size=self._batch_size)

    def _data_stream(self):
        """

        Generate Data Streams using ``keras.preprocessing.ImageDataGenerator()``.

        :raises: ValueError if there are asymmetries between the 'train' and 'validation'
                 subdirectories in ``self._data_path``.
        """
        # Create Data Streams
        self._train_gen()
        self._val_gen()

        # Update
        train_classes = self._train_generator.class_indices
        val_classes = self._validation_generator.class_indices

        def set_diff(a, b):
            """Return True if there are things in b not in a, else False."""
            return len(set(b) - set(a)) > 0

        # Run Check
        c = [(train_classes, val_classes, "train", "validation"), (val_classes, train_classes, "train", "validation")]
        for (i, j, k, l) in c:
            if set_diff(i, j):
                raise ValueError("the `{0}` folder is missing the following"
                                 " folders found in '{1}': {2}.".format(k, l, ", ".join(map(str, set(j) - set(i)))))

        self._data_classes = train_classes.class_indices

    def conv_net(self, loss='categorical_crossentropy', optimizer='rmsprop', metrics=('accuracy')):
        """

        Define and Compile the Image Recognition Convolutional Neural Network.

        :param loss: Loss function. Defaults to 'categorical_crossentropy'.
                     See: ``keras.models.Sequential()``.
        :type loss: ``str``
        :param optimizer: Optimizer name. Defaults to `rmsprop`.
                          See: ``keras.models.Sequential()``.
        :type optimizer: ``str``
        :param metrics: Metrics to evaluate. Defaults to ['accuracy'].
                        See: ``keras.models.Sequential()``.
        :type metrics: ``tuple``
        """
        if self._data_path is not None:
            self._data_stream()

        # Get the number of classes
        nb_classes = len(self._data_classes.keys()) if self._data_classes is not None else self._nb_classes_default

        # Define the Model
        self.model = Sequential()

        # Convolution layers to generate features
        # for the fully connected layers below.
        self.model.add(Convolution2D(32, 3, 3
                                     , input_shape=(3, self._img_shape[0], self._img_shape[1])
                                     , dim_ordering=self._dim_ordering))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=self._dim_ordering))

        self.model.add(Convolution2D(32, 3, 3, dim_ordering=self._dim_ordering))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=self._dim_ordering))

        self.model.add(Convolution2D(64, 3, 3, dim_ordering=self._dim_ordering))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=self._dim_ordering))

        # Fully Connected Layers, i.e., a standard
        # neural net being fed features from above.
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        # Output
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('sigmoid'))

        # Compilation
        self.model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics))

    def _model_existence_check(self, first_format, second_format):
        """

        Raises appropriate AttributeError based on content in which an undefined model was encountered.

        :param first_format: action the model
        :type first_format: ``str``
        :param second_format: name of an ``ImageRecognitionCNN`` method.
        :type second_format: ``str``
        :raises: AttributeError composed from `first_format` and `second_format`.
        """
        if self.model is None:
            raise AttributeError("The model cannot be {0} until `ImageRecognitionCNN().{1}()` "
                                 "has been called.".format(first_format, second_format))

    def fit(self, nb_epoch):
        """

        Fit the model to the training data and run a validation.

        :param nb_epoch: number of iterations. See: ``keras.models.Sequential()``. Defaults to 10.
        :type nb_epoch: ``int``
        :raises: AttributeError if `ImageRecognitionCNN().conv_net()` is yet to be called.
        """
        self._model_existence_check("fit and validated", "conv_net")
        self.model.fit_generator(generator=self._train_generator
                                 , samples_per_epoch=self._train_generator.nb_sample
                                 , nb_epoch=nb_epoch
                                 , validation_data=self._validation_generator
                                 , nb_val_samples=self._validation_generator.nb_sample)

    def save(self, path):
        """

        Save the weights from a trained model.

        :param path: path to save the data to. See: ``keras.models.Sequential()``.
        :type path: ``str``
        :raises: AttributeError if `ImageRecognitionCNN().fit_gen()` is yet to be called.
        """
        self._model_existence_check("saved", "fit_gen")
        self.model.save(path, overwrite=False)

    def load(self, path):
        """

        Saves the weights for the ConvNet.

        :param path: path to save the data to. See: ``keras.models.Sequential()``.
        :type path: ``str``
        :raises: AttributeError if `ImageRecognitionCNN().conv_net()` is yet to be called.
        """
        self._model_existence_check("loaded", "conv_net")
        self.model.load(path)










































