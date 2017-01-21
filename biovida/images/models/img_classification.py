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

from biovida.images.image_tools import load_and_scale_imgs

from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# Problem: ValueError: Negative dimension size caused by subtracting 2 from 1
# Solution: replace "tf" with "th" in ~/.keras/keras.json.
# Note: `MaxPooling2D` has a `dim_ordering` param which can do the same thing.
# When deployed, `dim_ordering` should equal 'th' to avoid users having this problem.
# The same goes to all function below which have this param.


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
    :param rescale: Defaults to 1/255. See: ``keras.preprocessing.image.ImageDataGenerator()``.
    :type rescale: ``float``
    :param shear_range: Defaults to 0.1. See: ``keras.preprocessing.image.ImageDataGenerator()``.
    :type shear_range: ``float``
    :param zoom_range: Defaults to 0.35. See: ``keras.preprocessing.image.ImageDataGenerator()``.
    :type zoom_range: ``float``
    :param horizontal_flip: See: ``keras.preprocessing.image.ImageDataGenerator()``.
    :type horizontal_flip: ``bool``
    :param batch_size: Samples to propagate through the model.
                       See: ``keras.preprocessing.ImageDataGenerator().flow_from_directory()``.
                       Defaults to 32.
    :type batch_size: ``int``
    """

    def __init__(self
                 , data_path=None
                 , img_shape=(150, 150)
                 , rescale=1/255.0
                 , shear_range=0.1
                 , zoom_range=0.35
                 , horizontal_flip=True
                 , batch_size=4):
        self._data_path = data_path
        self.img_shape = img_shape
        self.rescale = rescale
        self._shear_range = shear_range
        self._zoom_range = zoom_range
        self._horizontal_flip = horizontal_flip
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
            """Returns `True` if there are things in b not in a, else `False`."""
            return len(set(b) - set(a)) > 0

        # Check for a mismatch of folders between 'train' and 'validation'.
        chk = [(train_classes, val_classes, "train", "validation"), (val_classes, train_classes, "validation", "train")]
        for (i, j, k, l) in chk:
            if set_diff(i, j):
                raise ValueError("the `{0}` folder is missing the following"
                                 " folders found in '{1}': {2}.".format(k, l, ", ".join(map(str, set(j) - set(i)))))

        self.data_classes = train_classes

    def convnet(self, loss='categorical_crossentropy', optimizer='rmsprop', metrics=('accuracy',)):
        """

        Define and Compile the Image Recognition Convolutional Neural Network.


        :param loss: Loss function. Defaults to 'categorical_crossentropy'.
                     See: ``keras.models.Sequential()``.
        :type loss: ``str``
        :param optimizer: Optimizer name. Defaults to `rmsprop`.
                          See: ``keras.models.Sequential()``.
        :type optimizer: ``str``
        :param metrics: Metrics to evaluate. Defaults to ('accuracy',).
                        Note: if round braces are used, it MUST contain a comma (to make it a tuple).
                        See: ``keras.models.Sequential()``.
        :type metrics: ``tuple``
        """
        if self._data_path is not None:
            self._data_stream()

        # Get the number of classes
        nb_classes = len(self.data_classes.keys())

        # Define the Model
        self.model = Sequential()

        # Convolution layers to generate features
        # for the fully connected layers below.
        self.model.add(Convolution2D(32, 3, 3
                                     , input_shape=(3, self.img_shape[0], self.img_shape[1])
                                     , dim_ordering=self._dim_ordering, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=self._dim_ordering))

        self.model.add(Convolution2D(32, 3, 3, dim_ordering=self._dim_ordering, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=self._dim_ordering))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('sigmoid'))

        # Compilation
        self.model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics))

    def _model_existence_check(self, first_format, second_format, additional=''):
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
        :raises: AttributeError if ``ImageRecognitionCNN().convnet()`` is yet to be called.
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

        :param save_name:
        :type save_name: ``str``
        :param save_path:
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
        :raises: AttributeError if `ImageRecognitionCNN().fit_gen()` is yet to be called.
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

        :param path:
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
        :raises: AttributeError if a model is currently instantiated.
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

    def _prediction_labels(self, single_img_prediction, d):
        """

        :param single_img_prediction:
        :type single_img_prediction: list of ndarray arrays.
        :param d: data_classes_reversed
        :type d: ``dict``
        :return:
        """
        predictions = ((d[e], i) for e, i in enumerate(single_img_prediction))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def predict(self, list_of_images, status=True, verbose=False):
        """

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

        def status_bar(x): # ToDo: Not working properly
            return tqdm(x) if status else x

        number_ndarray = ['ndarray' in str(type(i)) for i in list_of_images]

        if all(number_ndarray):
            images = list_of_images
        elif any(number_ndarray):
            raise ValueError("Only some of the items in `list_of_images` we found to be `ndarrays`.")
        else:
            if verbose:
                print("\nPreparing Images for Neural Network...")
            images = load_and_scale_imgs(list_of_images, img_size=self.img_shape, status=status)

        if verbose:
            print("\nGenerating Predictions...")

        data_classes_reversed = {v: k for k, v in self.data_classes.items()}
        return [self._prediction_labels(i, data_classes_reversed) for i in status_bar(self.model.predict(images))]







































