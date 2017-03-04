"""

    Image Classification
    ~~~~~~~~~~~~~~~~~~~~

"""
# Resources Used:
#    - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#    - https://blog.rescale.com/neural-networks-using-keras-on-rescale/
#    - http://cs231n.github.io

# Imports
import os
import pickle
from tqdm import tqdm
from warnings import warn
from biovida.images._image_tools import load_and_scale_imgs

from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, merge
from keras.optimizers import RMSprop, SGD


# Problem: ValueError: Negative dimension size caused by subtracting 2 from 1
# Solution: replace "tf" with "th" in ~/.keras/keras.json.
# Note: `MaxPooling2D` has a `dim_ordering` param which can do the same thing.

# ToDo: it's not currently obvious if keras.preprocessing.ImageDataGenerator().flow_from_directory()
# performs the preprocessing described in https://arxiv.org/pdf/1409.1556.pdf


class ImageRecognitionCNN(object):
    """

    Keras Convolutional Neural Networks Interface.

    :param data_path: path to the directory with the subdirectories entitled 'train' and 'validation'.
                      This directory *must* have this structure. Defaults to ``None`` (to be use when loading
                      pre-computed weights).
    :type data_path: ``str``
    :param img_shape: the (height, width) to rescale the images to. Elements must be ``ints``. Defaults to ``(150, 150)``.
    :type img_shape: ``tuple`` or ``list``.
    :param rescale: See: ``keras.preprocessing.image.ImageDataGenerator()``. Defaults to 1/255.
    :type rescale: ``float``
    :param shear_range: See: ``keras.preprocessing.image.ImageDataGenerator()``. Defaults to 0.1.
    :type shear_range: ``float``
    :param zoom_range: See: ``keras.preprocessing.image.ImageDataGenerator()``. Defaults to 0.35.
    :type zoom_range: ``float``
    :param horizontal_flip: See: ``keras.preprocessing.image.ImageDataGenerator()``. Defaults to ``True``.
    :type horizontal_flip: ``bool``
    :param vertical_flip: See: ``keras.preprocessing.image.ImageDataGenerator()``. Defaults to ``True``.
    :type vertical_flip: ``bool``
    :param batch_size: Samples to propagate through the model.
                       See: ``keras.preprocessing.ImageDataGenerator().flow_from_directory()``.
                       Defaults to 4.
    :type batch_size: ``int``
    """

    def __init__(self
                 , data_path=None
                 , img_shape=(150, 150)
                 , rescale=1/255.0
                 , shear_range=0.05
                 , zoom_range=0.30
                 , horizontal_flip=True
                 , vertical_flip=False
                 , batch_size=1):
        self._data_path = data_path
        self.img_shape = img_shape
        self.rescale = rescale
        self._shear_range = shear_range
        self._zoom_range = zoom_range
        self._horizontal_flip = horizontal_flip
        self._vertical_flip = vertical_flip
        self._batch_size = batch_size

        # Define data location
        if self._data_path is not None:
            self._train_data_dir = os.path.join(self._data_path, "train")
            self._validation_data_dir = os.path.join(self._data_path, "validation")

        # Data Streams
        self._train_generator = None
        self._validation_generator = None

        # The model itself
        self.model = None
        self.data_classes = None

    def _train_gen(self):
        """

        Use of ``keras.preprocessing.image.ImageDataGenerator()`` to generate training stream.

        """
        # Train augmentation configuration
        train_datagen = ImageDataGenerator(rescale=self.rescale
                                           , shear_range=self._shear_range
                                           , zoom_range=self._zoom_range
                                           , vertical_flip=self._vertical_flip
                                           , horizontal_flip=self._horizontal_flip)

        # Indefinitely generate batches of augmented train image data
        self._train_generator = train_datagen.flow_from_directory(directory=self._train_data_dir,
                                                                  target_size=self.img_shape,
                                                                  class_mode='categorical',
                                                                  batch_size=self._batch_size)

    def _val_gen(self):
        """

        Use of ``keras.preprocessing.image.ImageDataGenerator()`` to generate validation stream.

        """
        # Test augmentation configuration
        validation_datagen = ImageDataGenerator(rescale=self.rescale)

        # This is a similar generator, for validation data
        self._validation_generator = validation_datagen.flow_from_directory(directory=self._validation_data_dir,
                                                                            target_size=self.img_shape,
                                                                            class_mode='categorical',
                                                                            batch_size=self._batch_size)

    def _data_stream(self):
        """

        Generate Data Streams using ``keras.preprocessing.ImageDataGenerator()``.

        :raises: ``ValueError`` if there are asymmetries between the 'train' and 'validation'
                 subdirectories in ``self._data_path``.
        """
        # Create Data Streams
        self._train_gen()
        self._val_gen()

        # Update
        train_classes = self._train_generator.class_indices
        val_classes = self._validation_generator.class_indices

        def set_diff(a, b):
            """Returns `True` if there are things in b not in a, else `False`."""
            return len(set(b) - set(a)) > 0

        # Check for a mismatch of folders between 'train' and 'validation'.
        chk = [(train_classes, val_classes, "train", "validation"), (val_classes, train_classes, "validation", "train")]
        for (i, j, k, l) in chk:
            if set_diff(i, j):
                raise ValueError("the `{0}` folder is missing the following"
                                 " folders found in '{1}': {2}.".format(k, l, ", ".join(map(str, set(j) - set(i)))))

        self.data_classes = train_classes

    def _impose_vgg_img_reqs(self):
        """

        This method imposes the 224x224 image size requirement made by VGG 19.

        """
        self.img_shape = list(self.img_shape)
        if self.img_shape[0] != 224:
            warn("{0} is an invalid image height for vgg_19. Falling back 224.".format(self.img_shape[0]))
            self.img_shape[0] = 224
        if self.img_shape[1] != 224:
            warn("{0} is an invalid image width for vgg_19. Falling back to 224.".format(self.img_shape[1]))
            self.img_shape[1] = 224
        self.img_shape = tuple(self.img_shape)

    def _alex_net(self, nb_classes, output_layer_activation):
        """

        Sources:
        -------
        1. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

        2. https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py

        Copyright (c) 2016 Heuritech

        Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
        documentation files (the "Software"), to deal in the Software without restriction, including without limitation
        the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
        and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all copies or substantial portions
        of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
        TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
        THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
        CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
        DEALINGS IN THE SOFTWARE.

        :param nb_classes: number of neuron in the output layer (which equals the number of classes).
        :type nb_classes: ``int``
        :param output_layer_activation: the activation function to use on the output layer. See: https://keras.io/activations/#available-activations. Defaults to 'sigmoid'.
        :type output_layer_activation: ``str``
        """
        try:
            from convnetskeras.customlayers import crosschannelnormalization, splittensor
        except ImportError:
            raise ImportError("This method requires the 'convnets-keras' library.\n"
                              "This library can be installed by running the following command from the command line:\n"
                              "$ pip install git+git://github.com/heuritech/convnets-keras@master")

        inputs = Input(shape=(3, self.img_shape[0], self.img_shape[1]))

        conv_1 = Convolution2D(96, 11, 11,
                               subsample=(4, 4),
                               activation='relu',
                               name='conv_1')(inputs)

        conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
        conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
        conv_2 = ZeroPadding2D((2, 2))(conv_2)
        conv_2 = merge([Convolution2D(128, 5, 5, activation="relu", name='conv_2_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_2)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

        conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
        conv_3 = crosschannelnormalization()(conv_3)
        conv_3 = ZeroPadding2D((1, 1))(conv_3)
        conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

        conv_4 = ZeroPadding2D((1, 1))(conv_3)
        conv_4 = merge([Convolution2D(192, 3, 3, activation="relu", name='conv_4_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_4)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

        conv_5 = ZeroPadding2D((1, 1))(conv_4)
        conv_5 = merge([Convolution2D(128, 3, 3, activation="relu", name='conv_5_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_5)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")
        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(nb_classes, name='dense_3')(dense_3)
        prediction = Activation(output_layer_activation, name=output_layer_activation)(dense_3)

        self.model = Model(input=inputs, output=prediction)

    def _vgg_19(self, nb_classes, output_layer_activation):
        """

        Keras Implementation of the VGG_19 Model

        SOURCES:
        -------

        1. Model Authors:
           Very Deep Convolutional Networks for Large-Scale Image Recognition
           K. Simonyan, A. Zisserman
           arXiv:1409.1556

        2. Keras Implementation: https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d#file-vgg-19_keras-py

        :param nb_classes: number of neuron in the output layer (which equals the number of classes).
        :type nb_classes: ``int``
        :param output_layer_activation: the activation function to use on the output layer. See: https://keras.io/activations/#available-activations. Defaults to 'sigmoid'.
        :type output_layer_activation: ``str``
        """
        self.model = Sequential()

        self.model.add(ZeroPadding2D((1, 1), input_shape=(3, self.img_shape[0], self.img_shape[1])))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes, activation=output_layer_activation))

    def _default_model(self, nb_classes, output_layer_activation):
        """

        The most simple model in this class.

        :param nb_classes: number of neuron in the output layer (which equals the number of classes).
        :type nb_classes: ``int``
        :param output_layer_activation: the activation function to use on the output layer. See: https://keras.io/activations/#available-activations. Defaults to 'sigmoid'.
        :type output_layer_activation: ``str``
        """
        self.model = Sequential()
        self.model.add(Convolution2D(32, 3, 3
                                     , input_shape=(3, self.img_shape[0], self.img_shape[1])
                                     , activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(32, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation(output_layer_activation))

    def convnet(self,
                model_to_use='default',
                loss='binary_crossentropy',
                optimizer='default',
                metrics=('accuracy',),
                output_layer_activation='sigmoid'):
        """

        Define and Compile the Image Recognition Convolutional Neural Network.

        :param model_to_use: one of: 'default', 'vgg19', 'alex_net'. Defaults to 'default'.

            - 'default': a relatively simple sequential model with two convolution layers (each followed by 2x2 max pooling); one hidden layer and 0.5 drop out.

            - 'alex_net': the 2012 'AlexNet' model.

            - 'vgg19': the VGG 19 model.

        :type model_to_use: ``str``
        :param loss: Loss function. Defaults to 'categorical_crossentropy'.
                     See: ``keras.models.Sequential()``.
        :type loss: ``str``
        :param optimizer: Optimizer name. Defaults to 'default', which will use RMSprop with learning rate = ``0.0001``.
                          See: ``keras.models.Sequential()``.
        :type optimizer: ``str`` or ``keras.optimizers``
        :param metrics: Metrics to evaluate. Defaults to ('accuracy',).
                        Note: if round braces are used, it MUST contain a comma (to make it a tuple).
                        See: ``keras.models.Sequential()``.
        :type metrics: ``tuple``
        :param output_layer_activation: the activation function to use on the output layer. See: https://keras.io/activations/#available-activations. Defaults to 'sigmoid'.
        :type output_layer_activation: ``str``
        """
        if model_to_use == 'vgg19':
            self._impose_vgg_img_reqs()

        if self._data_path is not None:
            self._data_stream()

        # Get the number of classes
        nb_classes = len(self.data_classes.keys())

        # Define the Model
        if model_to_use == 'default':
            self._default_model(nb_classes, output_layer_activation=output_layer_activation)
        elif model_to_use == 'alex_net':
            self._alex_net(nb_classes, output_layer_activation=output_layer_activation)
        elif model_to_use == 'vgg19':
            self._vgg_19(nb_classes, output_layer_activation=output_layer_activation)
        else:
            raise ValueError("'{0}' is an invalid value for `model_to_use`.".format(model_to_use))

        # Define optimizer
        if optimizer == 'default':
            optimizer_to_pass = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
        elif optimizer == 'vgg19':
            optimizer_to_pass = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optimizer_to_pass = optimizer

        # Compilation
        self.model.compile(loss=loss, optimizer=optimizer_to_pass, metrics=list(metrics))

    def _model_existence_check(self, first_format, second_format, additional=''):
        """

        Raises appropriate AttributeError based on content in which an undefined model was encountered.

        :param first_format: action the model
        :type first_format: ``str``
        :param second_format: name of an ``ImageRecognitionCNN`` method.
        :type second_format: ``str``
        :raises: ``AttributeError`` composed from `first_format` and `second_format`.
        """
        if self.model is None:
            raise AttributeError("The model cannot be {0} until `ImageRecognitionCNN().{1}()` "
                                 "has been called.{2}".format(first_format, second_format, additional))

    def fit(self, nb_epoch=10, min_delta=0.1, patience=3):
        """

        Fit the model to the training data and run a validation.

        :param nb_epoch: number of epochs. See: ``keras.models.Sequential()``. Defaults to 10.
        :type nb_epoch: ``int``
        :param min_delta: see ``keras.callbacks.EarlyStopping()``.
        :type min_delta: ``float``
        :param patience: see ``keras.callbacks.EarlyStopping()``.
        :type patience: ``int``
        :raises: ``AttributeError`` if ``ImageRecognitionCNN().convnet()`` is yet to be called.
        """
        self._model_existence_check("fit and validated", "convnet")

        if not isinstance(nb_epoch, int):
            raise ValueError("`nb_epoch` must be an intiger.")

        # Define callbacks
        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1)

        self.model.fit_generator(generator=self._train_generator
                                 , samples_per_epoch=self._train_generator.nb_sample
                                 , nb_epoch=nb_epoch
                                 , validation_data=self._validation_generator
                                 , nb_val_samples=self._validation_generator.nb_sample
                                 , callbacks=[early_stop])

    def _support_save_data(self, save_name, save_path):
        """

        Supporting Data For the Model.

        :param save_name: see ``save()``
        :type save_name: ``str``
        :param save_path: see ``save()``
        :type save_path: ``str``
        """
        data = [self._data_path,
                self.img_shape,
                self.rescale,
                self._shear_range,
                self._zoom_range,
                self._horizontal_flip,
                self._batch_size,
                self.data_classes]

        # Pickle the `data` dictionary.
        save_location = os.path.join(save_path, "{0}_support.p".format(save_name))
        pickle.dump(data, open(save_location, "wb"))

    def save(self, save_name, path=None, overwrite=False):
        """

        Save the weights from a trained model.

        :param save_name: name of the file. Do not include the '.h5' extension as it
                     will be added automatically.
        :type save_name: ``str``
        :param path: path to save the data to. See: ``keras.models.Sequential()``.
        :type path: ``str``
        :param overwrite: overwrite the existing copy of the data
        :type overwrite: ``bool``
        :raises: ``AttributeError`` if ``ImageRecognitionCNN().fit()`` is yet to be called.
        """
        self._model_existence_check("saved", "fit",  " Alternatively, you can call .load().")
        save_path = self._data_path if (path is None and self._data_path is not None) else path

        # Save the supporting data
        self._support_save_data(save_name, save_path)

        # Save the Model itself
        self.model.save(os.path.join(save_path, "{0}.h5".format(save_name)), overwrite=overwrite)

    def _support_load_data(self, path):
        """

        Loads supporting data saved along with the model

        :param path: see ``load()``.
        :type path: ``str``
        """
        load_location = "{0}_support.p".format(path[:-3])
        data = pickle.load(open(load_location, "rb"))

        self._data_path = data[0]
        self.img_shape = data[1]
        self.rescale = data[2]
        self._shear_range = data[3]
        self._zoom_range = data[4]
        self._horizontal_flip = data[5]
        self._batch_size = data[6]
        self.data_classes = data[7]

    def load(self, path, override_existing=False, default_model_load=False):
        """

        Load a model from disk.

        :param path: path to save the data to.See: ``keras.models.Sequential()``.
        :type path: ``str``
        :param override_existing: If True and a model has already been instantiated, override this replace this model.
                                  Defaults to ``False``.
        :type override_existing: ``bool``
        :param default_model_load: load the default model if ``ImageRecognitionCNN().convnet()`` has not been called.
                                   Defaults to ``False``.
        :type default_model_load: ``bool``
        :raises: ``AttributeError`` if a model is currently instantiated.
        """
        if self.model is not None and override_existing is not True:
            raise AttributeError("A model is currently instantiated.\n"
                                 "Set `override_existing` to `True` to replace the existing model.")

        if default_model_load:
            self.convnet()

        # Load the supporting data
        self._support_load_data(path)

        # Load the Model
        self.model = load_model(path)

    def _prediction_labels(self, single_img_prediction):
        """

        Convert a single prediction into a human-readable list of tuples.

        :param single_img_prediction: see ``predict()``
        :type single_img_prediction: list of ndarray arrays.
        :return: a list of tuples where the elements are of the form ``(label, P(label))``
        :rtype: ``list``
        """
        data_classes_reversed = {v: k for k, v in self.data_classes.items()}
        predictions = ((data_classes_reversed[e], i) for e, i in enumerate(single_img_prediction))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def predict(self, list_of_images, status=True, verbose=False):
        """

        Generate Predictions for a list of images.

        :param list_of_images: a list of paths (strings) to images or ``ndarrays``.
        :param status: True for a tqdm status bar; False for no status bar. Defaults to True.
        :type status: ``bool``
        :param verbose: if True, print updates. Defaults to False
        :type verbose: ``bool``
        :return: a list of lists with tuples of the form (name, probabability). Defaults to False.
        :rtype: ``list``
        """
        if self.model is None:
            raise AttributeError("Predictions cannot be made until a model is loaded or trained.")

        def status_bar(x): # ToDo: Not working properly (see: https://github.com/bstriner/keras-tqdm)
            return tqdm(x) if status else x

        number_ndarray = ['ndarray' in str(type(i)) for i in list_of_images]

        if all(number_ndarray):
            images = list_of_images
        elif any(number_ndarray):
            raise ValueError("Only some of the items in `list_of_images` we found to be `ndarrays`.")
        else:
            if verbose:
                print("\n\nPreparing Images for Neural Network...")
            images = load_and_scale_imgs(list_of_images, img_size=self.img_shape, status=status, grayscale_first=True)

        if verbose:
            print("\n\nGenerating Predictions...")
        return [self._prediction_labels(i) for i in status_bar(self.model.predict(images))]







































