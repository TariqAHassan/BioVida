# coding: utf-8

"""

    Image Classification
    ~~~~~~~~~~~~~~~~~~~~

"""
import os
import pickle
from warnings import warn

from biovida.support_tools.support_tools import tqdm
from biovida.images._image_tools import load_and_scale_images

# The model in `/_resources` was trained using Theano.
# TensorFlow cannot be used to load these weights until
# this issue (https://github.com/fchollet/keras/pull/5898) is resolved.
os.environ['KERAS_BACKEND'] = 'theano'

from keras import callbacks
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import (Convolution2D,
                          MaxPooling2D,
                          ZeroPadding2D,
                          concatenate,
                          GlobalAveragePooling2D,
                          Activation,
                          Dropout,
                          Flatten,
                          Dense,
                          Input,
                          merge)
from keras.optimizers import RMSprop, SGD

# Problem: ValueError: Negative dimension size caused by subtracting 2 from 1
# Solution: replace "tf" with "th" in ~/.keras/keras.json.
# Note: `MaxPooling2D` has a `dim_ordering` param which can do the same thing.

# ToDo: it's not currently obvious if keras.preprocessing.ImageDataGenerator().flow_from_directory()
# performs the preprocessing described in https://arxiv.org/pdf/1409.1556.pdf


class ImageClassificationCNN(object):
    """

    Keras Convolutional Neural Networks Interface.

    :param data_path: path to the directory with the subdirectories entitled 'train' and 'validation'.
                      This directory *must* have this structure. Defaults to ``None`` (to be use when loading
                      pre-computed weights).
    :type data_path: ``str``
    :param image_shape: the (height, width) to rescale the images to. Elements must be ``ints``. Defaults to ``(150, 150)``.
    :type image_shape: ``tuple`` or ``list``.
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

    def __init__(self,
                 data_path=None,
                 image_shape=(224, 224),
                 rescale=1/255.0,
                 shear_range=0.05,
                 zoom_range=0.30,
                 horizontal_flip=True,
                 vertical_flip=False,
                 batch_size=1):
        self._data_path = data_path
        self.image_shape = image_shape
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
        train_datagen = ImageDataGenerator(rescale=self.rescale,
                                           shear_range=self._shear_range,
                                           zoom_range=self._zoom_range,
                                           vertical_flip=self._vertical_flip,
                                           horizontal_flip=self._horizontal_flip)

        # Indefinitely generate batches of augmented train image data
        self._train_generator = train_datagen.flow_from_directory(directory=self._train_data_dir,
                                                                  target_size=self.image_shape,
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
                                                                            target_size=self.image_shape,
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

    def _impose_vgg_image_reqs(self):
        """

        This method imposes the 224x224 image size requirement made by VGG 19.

        """
        self.image_shape = list(self.image_shape)
        if self.image_shape[0] != 224:
            warn("{0} is an invalid image height for vgg_19. Falling back 224.".format(self.image_shape[0]))
            self.image_shape[0] = 224
        if self.image_shape[1] != 224:
            warn("{0} is an invalid image width for vgg_19. Falling back to 224.".format(self.image_shape[1]))
            self.image_shape[1] = 224
        self.image_shape = tuple(self.image_shape)

    def default(self, classes, output_layer_activation):
        """

        A Convolutional Neural Network with two convolution layers and ~5 million parameters.

        Sources:

        - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

        :param classes: number of neuron in the output layer (which equals the number of classes).
        :type classes: ``int``
        :param output_layer_activation: the activation function to use on the output layer. See: https://keras.io/activations/#available-activations. Defaults to 'sigmoid'.
        :type output_layer_activation: ``str``
        """
        self.model = Sequential()
        self.model.add(Convolution2D(32, (3, 3),
                                     input_shape=(3, self.image_shape[0], self.image_shape[1]),
                                     activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(classes))
        self.model.add(Activation(output_layer_activation))

    def _alex_net(self, classes, output_layer_activation):
        """

        .. warning:
        
            This is *NOT* stable with Keras 2.

        Sources:
        
        1. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

        2. https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py

        :param classes: number of neuron in the output layer (which equals the number of classes).
        :type classes: ``int``
        :param output_layer_activation: the activation function to use on the output layer. See: https://keras.io/activations/#available-activations. Defaults to 'sigmoid'.
        :type output_layer_activation: ``str``
        """
        # Copyright (c) 2016 Heuritech
        #
        # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
        # documentation files (the "Software"), to deal in the Software without restriction, including without limitation
        # the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
        # and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
        #
        # The above copyright notice and this permission notice shall be included in all copies or substantial portions
        # of the Software.
        #
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
        # TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
        # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
        # CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
        # DEALINGS IN THE SOFTWARE.
        #
        # Note: not stable with Keras 2

        try:
            from convnetskeras.customlayers import crosschannelnormalization, splittensor
        except ImportError:
            raise ImportError("This method requires the 'convnets-keras' library.\n"
                              "This library can be installed by running the following command from the command line:\n"
                              "$ pip install git+git://github.com/tariqahassan/convnets-keras@master")

        inputs = Input(shape=(3, self.image_shape[0], self.image_shape[1]))

        conv_1 = Convolution2D(96, (11, 11),
                               subsample=(4, 4),
                               activation='relu',
                               name='conv_1')(inputs)

        conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
        conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
        conv_2 = ZeroPadding2D((2, 2))(conv_2)
        conv_2 = merge([Convolution2D(128, (5, 5), activation="relu", name='conv_2_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_2)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

        conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
        conv_3 = crosschannelnormalization()(conv_3)
        conv_3 = ZeroPadding2D((1, 1))(conv_3)
        conv_3 = Convolution2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

        conv_4 = ZeroPadding2D((1, 1))(conv_3)
        conv_4 = merge([Convolution2D(192, 3, 3, activation="relu", name='conv_4_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_4)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

        conv_5 = ZeroPadding2D((1, 1))(conv_4)
        conv_5 = merge([Convolution2D(128, (3, 3), activation="relu", name='conv_5_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_5)
                           ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")
        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(classes, name='dense_3')(dense_3)
        prediction = Activation(output_layer_activation, name=output_layer_activation)(dense_3)

        self.model = Model(input=inputs, output=prediction)

    def squeezenet(self, classes, output_layer_activation):
        """

        Keras Implementation of the SqueezeNet Model.

        Sources:
        
        - Iandola, Forrest N., Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer. 
          'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size.' arXiv preprint 
          `arXiv:1602.07360 <https://arxiv.org/abs/1602.07360>`_ (2016). 

        - Implementation adapted from: `github.com/rcmalli/keras-squeezenet <https://github.com/rcmalli/keras-squeezenet>`_

        :param classes: number of neuron in the output layer (which equals the number of classes).
        :type classes: ``int``
        :param output_layer_activation: the activation function to use on the output layer. See: https://keras.io/activations/#available-activations. Defaults to 'sigmoid'.
        :type output_layer_activation: ``str``
        """
        # MIT License
        #
        # Copyright (c) 2016 Refikcanmalli
        #
        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:
        #
        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.
        #
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        #
        # Note: the model has been slightly modified. Namely, it has been refactored and
        # simplified by removing use of ImageNet weights.

        sq1x1 = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu = "relu_"

        def fire_module(model, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            if K.image_data_format() == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = 3

            model = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(model)
            model = Activation('relu', name=s_id + relu + sq1x1)(model)

            left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(model)
            left = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(model)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            model = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
            return model

        input_img = Input(shape=(3, self.image_shape[0], self.image_shape[1]))

        model = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_img)
        model = Activation('relu', name='relu_conv1')(model)
        model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(model)

        model = fire_module(model, fire_id=2, squeeze=16, expand=64)
        model = fire_module(model, fire_id=3, squeeze=16, expand=64)
        model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(model)

        model = fire_module(model, fire_id=4, squeeze=32, expand=128)
        model = fire_module(model, fire_id=5, squeeze=32, expand=128)
        model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(model)

        model = fire_module(model, fire_id=6, squeeze=48, expand=192)
        model = fire_module(model, fire_id=7, squeeze=48, expand=192)
        model = fire_module(model, fire_id=8, squeeze=64, expand=256)
        model = fire_module(model, fire_id=9, squeeze=64, expand=256)
        model = Dropout(0.5, name='drop9')(model)

        model = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(model)
        model = Activation('relu', name='relu_conv10')(model)
        model = GlobalAveragePooling2D()(model)
        output = Activation(output_layer_activation, name='loss')(model)

        self.model = Model(input_img, output, name='squeezenet')

    def vgg_19(self, classes, output_layer_activation):
        """

        Keras Implementation of the VGG19 Model.

        Sources:
        -------

        - Very Deep Convolutional Networks for Large-Scale Image Recognition
          K. Simonyan, A. Zisserman. `arXiv:1409.1556 <https://arxiv.org/abs/1409.1556>`_.
           
        - The Keras Implementation is available `here <https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d#file-vgg-19_keras-py>`_.

        :param classes: number of neuron in the output layer (which equals the number of classes).
        :type classes: ``int``
        :param output_layer_activation: the activation function to use on the output layer. See: https://keras.io/activations/#available-activations. Defaults to 'sigmoid'.
        :type output_layer_activation: ``str``
        """
        self.model = Sequential()

        self.model.add(ZeroPadding2D((1, 1), input_shape=(3, self.image_shape[0], self.image_shape[1])))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(classes, activation=output_layer_activation))

    def convnet(self,
                model_to_use='default',
                loss='binary_crossentropy',
                optimizer='default',
                metrics=('accuracy',),
                output_layer_activation='sigmoid'):
        """

        Define and Compile the Image Recognition Convolutional Neural Network.

        :param model_to_use: one of: 'default', 'squeezenet' or 'vgg19'. Defaults to 'default'.

            - 'default': a relatively simple sequential model with two convolution layers.

            - 'squeezenet': SqueezeNet model.

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
            self._impose_vgg_image_reqs()

        if self._data_path is not None:
            self._data_stream()

        classes = len(self.data_classes.keys())

        try:
            model_method = getattr(self, model_to_use)
        except:
            raise AttributeError("No such model: '{0}'.".format(model_to_use))

        model_method(classes, output_layer_activation=output_layer_activation)

        # Define optimizer
        if optimizer == 'default':
            optimizer_to_pass = RMSprop(lr=0.000025, rho=0.9, epsilon=1e-08, decay=0.0)
        elif optimizer == 'squeezenet':
            optimizer_to_pass = SGD(lr=0.001, decay=0.0002, momentum=0.9, nesterov=True)
        elif optimizer == 'vgg19':
            optimizer_to_pass = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optimizer_to_pass = optimizer

        self.model.compile(loss=loss, optimizer=optimizer_to_pass, metrics=list(metrics))

    def _model_existence_check(self, first_format, second_format, additional=''):
        """

        Raises appropriate AttributeError based on content in which an undefined model was encountered.

        :param first_format: action the model
        :type first_format: ``str``
        :param second_format: name of an ``ImageClassificationCNN`` method.
        :type second_format: ``str``
        :raises: ``AttributeError`` composed from `first_format` and `second_format`.
        """
        if self.model is None:
            raise AttributeError("The model cannot be {0} until `ImageClassificationCNN().{1}()` "
                                 "has been called.{2}".format(first_format, second_format, additional))

    def fit(self, epochs=10, min_delta=0.1, patience=3):
        """

        Fit the model to the training data and run a validation.

        :param epochs: number of epochs. See: ``keras.models.Sequential()``. Defaults to 10.
        :type epochs: ``int``
        :param min_delta: see ``keras.callbacks.EarlyStopping()``.
        :type min_delta: ``float``
        :param patience: see ``keras.callbacks.EarlyStopping()``.
        :type patience: ``int``
        :raises: ``AttributeError`` if ``ImageClassificationCNN().convnet()`` is yet to be called.
        """
        self._model_existence_check("fit and validated", "convnet")

        if not isinstance(epochs, int):
            raise ValueError("`epochs` must be an integer.")

        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1)

        self.model.fit_generator(generator=self._train_generator,
                                 steps_per_epoch=self._train_generator.samples,
                                 epochs=epochs,
                                 validation_data=self._validation_generator,
                                 validation_steps=self._validation_generator.samples,
                                 callbacks=[early_stop])

    def _support_save_data(self, save_name, save_path):
        """

        Supporting Data For the Model.

        :param save_name: see ``save()``
        :type save_name: ``str``
        :param save_path: see ``save()``
        :type save_path: ``str``
        """
        data = [self._data_path,
                self.image_shape,
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
        :raises: ``AttributeError`` if ``ImageClassificationCNN().fit()`` is yet to be called.
        """
        self._model_existence_check("saved", "fit", " Alternatively, you can call .load().")
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
        with open(load_location, "rb") as f:
            data = pickle.load(f)

        self._data_path = data[0]
        self.image_shape = data[1]
        self.rescale = data[2]
        self._shear_range = data[3]
        self._zoom_range = data[4]
        self._horizontal_flip = data[5]
        self._batch_size = data[6]
        self.data_classes = data[7]

    def load(self, path, override_existing=False, default_model_load=False):
        """

        Load a model from disk.
        
        .. note::
            
            This method expects an additional file ending in "_support.p".
            For example: if ``path='/your/path/my_model.h5'``, a file entitled
            ``'/your/path/my_model_support.p'`` is also expected.

        :param path: path to save the data to.See: ``keras.models.Sequential()``.
        :type path: ``str``
        :param override_existing: If True and a model has already been instantiated, override this replace this model.
                                  Defaults to ``False``.
        :type override_existing: ``bool``
        :param default_model_load: load the default model if ``ImageClassificationCNN().convnet()`` has not been called.
                                   Defaults to ``False``.
        :type default_model_load: ``bool``
        :raises: ``AttributeError`` if a model is currently instantiated.
        """
        if self.model is not None and override_existing is not True:
            raise AttributeError("A model is currently instantiated.\n"
                                 "Set `override_existing` to `True` to replace the existing model.")

        if default_model_load:
            self.convnet()

        self._support_load_data(path)
        self.model = load_model(path)

    def _prediction_labels(self, single_image_prediction):
        """

        Convert a single prediction into a human-readable list of tuples.

        :param single_image_prediction: see ``predict()``
        :type single_image_prediction: list of ndarray arrays.
        :return: a list of tuples where the elements are of the form ``(label, P(label))``
        :rtype: ``list``
        """
        data_classes_reversed = {v: k for k, v in self.data_classes.items()}
        predictions = ((data_classes_reversed[e], i) for e, i in enumerate(single_image_prediction))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def predict(self, list_of_images, desc=None, status=True):
        """

        Generate Predictions for a list of images.

        :param list_of_images: a list of paths (strings) to images or ``ndarrays``.
        :type list_of_images: ``list``
        :param desc: description for ``tqdm``.
        :type desc: ``str`` or ``None``
        :param status: True for a tqdm status bar; False for no status bar. Defaults to True.
        :type status: ``bool``
        :return: a list of lists with tuples of the form (name, probability). Defaults to False.
        :rtype: ``list``
        """
        if self.model is None:
            raise AttributeError("Predictions cannot be made until a model is loaded or trained.")

        is_ndarray = [type(i).__name__ == 'ndarray' for i in list_of_images]

        if all(is_ndarray):
            images = list_of_images
        elif any(is_ndarray):
            raise ValueError("Only some of the items in `list_of_images` we found to be `ndarrays`.")
        else:
            images = load_and_scale_images(list_of_images=list_of_images, image_size=self.image_shape,
                                           status=status, grayscale_first=True,
                                           desc='Preparing Images')

        # ToDo: tqdm not working properly (see: https://github.com/bstriner/keras-tqdm).
        return [self._prediction_labels(i) for i in tqdm(self.model.predict(images), desc=desc, disable=not status)]
