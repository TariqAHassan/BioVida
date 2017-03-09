"""

    Image Processing
    ~~~~~~~~~~~~~~~~

"""
# Imports
import os
import numpy as np
import pandas as pd
import pkg_resources
from PIL import Image
from tqdm import tqdm
from warnings import warn
from PIL import ImageStat
from scipy.misc import imread

# General tools
from biovida.support_tools.support_tools import items_null

# Tools form the image subpackage
from biovida.images._image_tools import load_and_scale_imgs

# Models
from biovida.images.models.border_detection import border_detection
from biovida.images.models.template_matching import robust_match_template
from biovida.images.models.image_classification import ImageRecognitionCNN

# Suppress Pandas' SettingWithCopyWarning
pd.options.mode.chained_assignment = None


class ImageProcessing(object):
    """

    This class is designed to allow easy analysis of cached image data.

    :param instance: an instance of the ``biovida.images.openi_interface.OpenInterface()`` class.
    :type instance: ``OpenInterface Class``
    :param db_to_extract: ``records_db`` or``cache_records_db``. Defaults to 'records_db'.
    :type db_to_extract: ``str``
    :param model_location: the location of the model for Convnet.
                      If `None`, the default model will be used. Defaults to ``None``.
    :type model_location: ``str``
    :param verbose: if ``True``, print additional details. Defaults to ``False``.
    :type verbose: ``bool``

    :var image_dataframe: this is the dataframe that was passed when instantiating the class and
                          contains a cache of all analyses run as new columns.
    """

    @staticmethod
    def _extract_db(instance, db_to_extract):
        """

        Extracts a database from the `instance` parameter of ImageProcessing().

        :param db_to_extract: see ``ImageProcessing()``.
        :type db_to_extract: ``str``
        :param instance: see ``ImageProcessing()``.
        :type instance: ``OpenInterface Class``
        :return: extract database
        :rtype: ``Pandas DataFrame``
        """
        if db_to_extract not in ('records_db', 'cache_records_db'):
            raise ValueError("`db_to_extract` must be one of: 'records_db', 'cache_records_db'.")

        extract = getattr(instance, db_to_extract)

        if isinstance(extract, pd.DataFrame):
            image_dataframe = extract.copy(deep=True)
            return image_dataframe
        else:
            raise TypeError("The '{0}' of `instance` must be of "
                            "type DataFrame, not: '{2}'.".format(db_to_extract, type(extract).__name__))

    def __init__(self, instance, db_to_extract='records_db', model_location=None, verbose=True):
        self._verbose = verbose

        if "OpeniInterface" != type(instance).__name__:
            raise ValueError("`instance` must be a `OpeniInterface` instance.")

        # Extract the records_db/cache_records_db database
        self.image_dataframe = self._extract_db(instance, db_to_extract)

        # Extract path to the MedPix Logo
        self._medpix_path = instance._created_img_dirs['medpix_logo']

        # Spin up tqdm
        tqdm.pandas("status")

        # Load the CNN
        self._ircnn = ImageRecognitionCNN()

        # Load the model weights and architecture.
        model_path = pkg_resources.resource_filename('biovida', 'images/_resources/visual_image_problems_model.h5')
        if model_location is None:
            self._ircnn.load(model_path, override_existing=True)
        elif not isinstance(model_location, str):
            raise ValueError("`model_location` must either be a string or `None`.")
        elif os.path.isfile(model_location):
            self._ircnn.load(model_path, override_existing=True)
        else:
            raise FileNotFoundError("'{0}' could not be located.".format(str(model_location)))

        # Load the visual image problems the model can detect
        self.model_classes = list(self._ircnn.data_classes.keys())

        # Container for images represented as `ndarrays`
        self._ndarrays_images = None

        # Switch to control verbosity
        self._print_update = False

    @staticmethod
    def _apply_status(x, status):
        """

        Applies a tqdm() progress bar to an an iterable (if `status` is True).

        :param x: any iterable data structure.
        :type x: ``iterable``
        :return: ``x`` wrapped in a tqdm status bar if ``status`` is True, else the object is returned 'as is'.
        :type: ``tqdm`` or ``any``
        """
        if status:
            return tqdm(x)
        else:
            return x

    def _pil_load(self, image_paths, convert_to_rgb, status):
        """

        Load an image from a list of paths using the ``PIL`` Library.

        :param image_paths: images paths to load.
        :type image_paths: ``list``, ``tuple`` or ``Pandas Series``
        :param convert_to_rgb: if True, convert the image to RGB.
        :type convert_to_rgb: ``bool``
        :param status: display status bar. Defaults to True.
        :type status: ``bool``
        :return: a list of a. PIL images or b. PIL images in tuples of the form (PIL Image, image path).
        :rtype: ``list``
        """
        def conversion(img):
            return (Image.open(img).convert('RGB'), img) if convert_to_rgb else (Image.open(img), img)
        return [conversion(i) for i in self._apply_status(image_paths, status)]

    def _ndarray_extract(self, zip_with_column=None, reload_override=False):
        """

        Loads images as `ndarrays` and flattens them.

        :param zip_with_column: a column from the `image_dataframe` to zip with the images. Defaults to None.
        :type zip_with_column: ``str``
        :param reload_override: if True, reload the images from disk. Defaults to False.
        :type reload_override: ``bool``
        :return: images as 2D ndarrays.
        :rtype: ``zip`` or ``list of ndarrays``
        """
        if self._ndarrays_images is None or reload_override:
            self._ndarrays_images = [imread(i, flatten=True) for i in self.image_dataframe['cached_images_path']]

        if zip_with_column is not None:
            return zip(*[self._ndarrays_images, self.image_dataframe[zip_with_column]])
        else:
            return self._ndarrays_images

    @staticmethod
    def _grayscale_img(img_path):
        """

        Computes whether or not an image is grayscale.
        See the `grayscale_analysis()` method for caveats.

        :param img_path: path to an image.
        :type img_path:``str``
        :return: ``True`` if grayscale, else ``False``.
        :rtype: ``bool``
        """
        # See: http://stackoverflow.com/q/23660929/4898004
        if img_path is None or items_null(img_path):
            return np.NaN
        img = Image.open(img_path)  # ToDo: find way to call only once inside this class (similar to _ndarray_extract())
        stat = ImageStat.Stat(img.convert("RGB"))
        return np.mean(stat.sum) == stat.sum[0]

    def grayscale_analysis(self, new_analysis=False):
        """

        Analyze the images to determine whether or not they are grayscale
        (uses the PIL image library).

        Note:
              - this tool is very conservative (very small amounts of 'color' will yield `False`).
              - the exception to the above rule is the *very rare* case of an image which even split
                of red, green and blue.

        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        """
        if self._verbose and self._print_update:
            print("\n\nStarting Grayscale Analysis...")

        if 'grayscale' not in self.image_dataframe.columns or new_analysis:
            self.image_dataframe['grayscale'] = self.image_dataframe['cached_images_path'].progress_map(
                self._grayscale_img, na_action='ignore')

    @staticmethod
    def _logo_analysis_out(analysis_results, output_params):
        """

        Decides the output for the ``logo_analysis`` function.
        If the bonding box is in an improbable location, NaN is returned.
        Otherwise, the bonding box, or some portion of it (i.e., the lower left) will be returned.

        :param analysis_results: the output of ``biovida.images.models.template_matching.robust_match_template()``.
        :type analysis_results: ``dict``
        :return: the output requested by the ``logo_analysis()`` method.
        :rtype: ``NaN``, ``dict`` or ``tuple``
        """
        # Unpack ``output_params``
        match_quality_threshold, x_greater_check, y_greater_check, return_full = output_params

        # Unpack ``analysis_results``
        bounding_box = analysis_results['bounding_box']
        match_quality = analysis_results['match_quality']
        base_img_shape = analysis_results['base_img_shape']

        # Check match quality.
        if bounding_box is None or match_quality < match_quality_threshold:
            return np.NaN

        # Check the box is in the top right (as defined by ``x_greater_check`` and ``y_greater_check``).
        if bounding_box['bottom_left'][0] < (base_img_shape[0] * x_greater_check) or \
                        bounding_box['bottom_left'][1] > (base_img_shape[1] * y_greater_check):
            return np.NaN

        if return_full:
            return bounding_box
        else:
            return bounding_box['bottom_left']

    def _logo_processor(self, robust_match_template_wrapper, output_params, status):
        """

        Performs the actual analysis for ``logo_analysis()``, searching for the
        MedPix logo in the images.

        :param robust_match_template_wrapper: wrapper generated inside of ``logo_analysis()``
        :type robust_match_template_wrapper: ``function``
        :param output_params: tuple of the form:

                        ``(match_quality_threshold, xy_position_threshold[0], xy_position_threshold[1], return_full)``

        :type output_params: ``tuple``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        :return: a list of dictionaries or tuples specifying all, or part of (i.e., lower left) the bonding box
                 for the pattern in the base image.
        :rtype: ``list of dicts`` or ``list of tuples``
        """
        results = list()

        # Use images in the dataframe represented as ndarrays, along with
        # the journal title (to check for their source being medpix).
        to_analyze = self._ndarray_extract(zip_with_column='journal_title')

        for img, journal in self._apply_status(to_analyze, status):
            if 'medpix' not in str(journal).lower():
                results.append(np.NaN)
            else:
                analysis_results = robust_match_template_wrapper(img)
                current = self._logo_analysis_out(analysis_results, output_params)
                results.append(current)

        return results

    def logo_analysis(self,
                      match_quality_threshold=0.25,
                      xy_position_threshold=(1 / 3.0, 1 / 2.5),
                      base_resizes=(0.5, 2.5, 0.1),
                      end_search_threshold=0.875,
                      base_img_cropping=(0.15, 0.5),
                      return_full=False,
                      new_analysis=False,
                      status=True):
        """

        Search for the MedPix Logo. If located, with match quality above match_quality_threshold,
        populate the corresponding row of the 'medpix_logo_lower_left' column in the 'image_dataframe'
        with its a full bonding box if ``return_full=True``, otherwise only the lower left corner.

        :param match_quality_threshold: the minimum match quality required to accept the match.
                                        See: ``skimage.feature.match_template()`` for more information.
        :type match_quality_threshold: ``float``
        :param xy_position_threshold: tuple of the form: (x_greater_check, y_greater_check).
                                      For instance the default (``(1/3.0, 1/2.5)``) requires that the
                                      x position of the logo is greater than 1/3 of the image's width
                                      and less than 1/2.5 of the image's height.
        :type xy_position_threshold: ``tuple``
        :param base_resizes: See: ``biovida.images.models.template_matching.robust_match_template()``.
        :type base_resizes: ``tuple``
        :param end_search_threshold: See: ``biovida.images.models.template_matching.robust_match_template()``.
        :type end_search_threshold: ``float``
        :param base_img_cropping: See: ``biovida.images.models.template_matching.robust_match_template()``
        :type base_img_cropping: ``tuple``
        :param return_full: if ``True``, return a dictionary with the location of all four corners for the
                            logo's bounding box. Otherwise, only the bottom left corner will be returned.
                            Defaults to ``False``.
                            Note: ``True`` **cannot** be used in conjunction with 'auto' methods in this class.
        :type return_full: ``bool``
        :param new_analysis: rerun the analysis if it has already been computed.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        """
        # Note: this method wraps ``biovida.images.models.template_matching.robust_match_template()``.
        if 'medpix_logo_lower_left' in self.image_dataframe.columns and not new_analysis:
            return None

        if self._verbose and self._print_update:
            print("\n\nStarting Logo Analysis...")

        # Package Params
        output_params = (match_quality_threshold, xy_position_threshold[0], xy_position_threshold[1], return_full)

        # Load the Pattern. ToDo: Allow for non MedPix logos logos.
        medpix_template_img = imread(self._medpix_path, flatten=True)

        def robust_match_template_wrapper(img):
            return robust_match_template(pattern_img=medpix_template_img,
                                         base_img=img,
                                         base_resizes=base_resizes,
                                         end_search_threshold=end_search_threshold,
                                         base_img_cropping=base_img_cropping)

        # Run the algorithm searching for the medpix logo in the base image
        results = self._logo_processor(robust_match_template_wrapper, output_params, status)

        # Update dataframe with the (x, y) values of the lower left corner of the logo's bonding box.
        self.image_dataframe['medpix_logo_lower_left'] = results

    def border_analysis(self,
                        signal_strength_threshold=0.25,
                        min_border_separation=0.15,
                        lower_bar_search_space=0.9,
                        new_analysis=False,
                        status=True):
        """

        Wrapper for ``biovida.images.models.border_detection.border_detection()``.

        :param signal_strength_threshold: see ``biovida.images.models.border_detection()``.
        :type signal_strength_threshold: ``float``
        :param min_border_separation: see ``biovida.images.models.border_detection()``.
        :type min_border_separation: ``float``
        :param lower_bar_search_space: see ``biovida.images.models.border_detection()``.
        :type lower_bar_search_space: ``float``
        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        """
        if all(x in self.image_dataframe.columns for x in ['hbar', 'hborder', 'vborder']) and not new_analysis:
            return None

        if self._verbose and self._print_update:
            print("\n\nStarting Border Analysis...")

        def ba_func(image):
            return border_detection(image,
                                    signal_strength_threshold,
                                    min_border_separation,
                                    lower_bar_search_space,
                                    report_signal_strength=False,
                                    rescale_input_ndarray=True)

        to_analyze = self._ndarray_extract()

        # Run the analysis.
        border_analysis = [ba_func(i) for i in self._apply_status(to_analyze, status)]

        # Convert to a dataframe and return
        ba_df = pd.DataFrame(border_analysis).fillna(np.NaN)

        # Update datafame
        self.image_dataframe['hbar'] = ba_df['hbar']
        self.image_dataframe['hborder'] = ba_df['hborder']
        self.image_dataframe['vborder'] = ba_df['vborder']

    @staticmethod
    def _h_crop_top_decision(x):
        """

        Choose lowest horizontal cropping point.
        Solves: upper 'hborder' vs 'medpix_logo_lower_left'.

        :param x: data passed through Pandas ``DataFrame.apply()`` method.
        :type x: ``Pandas Object``
        :return: the lowest crop location.
        :rtype: ``int`` or ``float``
        """
        # Note: hborder = [top, lower]; medpix_logo_lower_left = [x, y].
        cols = ('hborder', 'medpix_logo_lower_left')
        crop_candidates = [x[i][0] if i == 'hborder' else x[i][1] for i in cols if not items_null(x[i])]
        return max(crop_candidates) if len(crop_candidates) else np.NaN

    @staticmethod
    def _h_crop_lower_decision(x):
        """

        Chose the highest cropping point for the image's bottom.
        Solves: lower 'hborder' vs 'hbar'.

        :param x: data passed through Pandas ``DataFrame.apply()`` method.
        :type x: ``Pandas Object``
        :return: the highest crop location.
        :rtype: ``int`` or ``float``
        """
        # Note: hborder = [top, lower]; hbar = int.
        lhborder = [x['hborder'][1]] if not items_null(x['hborder']) else []
        hbar = [x['hbar']] if not items_null(x['hbar']) else []
        crop_candidates = lhborder + hbar
        return min(crop_candidates) if len(crop_candidates) else np.NaN

    def crop_decision(self, new_analysis=False):
        """

        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        """
        if all(x in self.image_dataframe.columns for x in ['upper_crop', 'lower_crop']) and not new_analysis:
            return None

        for i in ('medpix_logo_lower_left', 'hborder', 'hbar'):
            if i not in self.image_dataframe.columns:
                raise KeyError("The `image_dataframe` does not contain the\nfollowing required column: '{0}'.\n"
                               "Please execute the corresponding analysis method to generate it.".format(i))

        # Compute Crop location
        self.image_dataframe['upper_crop'] = self.image_dataframe.apply(self._h_crop_top_decision, axis=1)
        self.image_dataframe['lower_crop'] = self.image_dataframe.apply(self._h_crop_lower_decision, axis=1)

    @staticmethod
    def _apply_cropping(cached_images_path,
                        lower_crop,
                        upper_crop,
                        vborder,
                        return_as_array=True,
                        convert_to_rgb=True):
        """

        Applies cropping to a specific image.

        :param cached_images_path: path to the image
        :type cached_images_path: ``str``
        :param lower_crop: row of the column produced by the ``crop_decision()`` method.
        :type lower_crop: ``int`` (can be ``float`` if the column contains NaNs)
        :param upper_crop: row of the column produced by the ``crop_decision()`` method.
        :type upper_crop: ``int`` (can be ``float`` if the column contains NaNs)
        :param vborder: yeild of the ``border_analysis()`` method
        :type vborder: ``int`` (can be ``float`` if the column contains NaNs)
        :param return_as_array: if True, convert the PIL object to an ``ndarray``. Defaults to True.
        :type return_as_array: ``bool``
        :return: the cropped image as either a PIL image or 2D ndarray.
        :rtype: ``PIL`` or ``2D ndarray``
        """
        # Load the image
        converted_image = Image.open(cached_images_path)

        # Horizontal Cropping
        if not items_null(lower_crop):
            w, h = converted_image.size
            converted_image = converted_image.crop((0, 0, w, int(lower_crop)))
        if not items_null(upper_crop):
            w, h = converted_image.size
            converted_image = converted_image.crop((0, int(upper_crop), w, h))

        # Vertical Cropping
        if not items_null(vborder):
            w, h = converted_image.size
            converted_image = converted_image.crop((int(vborder[0]), 0, int(vborder[1]), h))

        if convert_to_rgb:
            img_to_save = converted_image.convert("RGB")
        else:
            img_to_save = converted_image

        if return_as_array:
            return np.asarray(img_to_save)
        else:
            return img_to_save

    def _cropper(self, data_frame=None, return_as_array=True, convert_to_rgb=True, status=True):
        """

        Uses `_apply_cropping()` to apply cropping to images in a dataframe.

        :param data_frame: a dataframe with 'cached_images_path', 'lower_crop', 'upper_crop' and 'vborder' columns.
                          if None ``image_dataframe`` is used.
        :type data_frame: ``None`` or ``Pandas DataFrame``
        :param return_as_array: if True, convert the PIL object to an ``ndarray``. Defaults to True.
        :type return_as_array: ``bool``
        :param convert_to_rgb: if True, use the PIL library to convert the images to RGB. Defaults to False.
        :type convert_to_rgb: ``bool``
        :param status: display status bar. Defaults to True.
        :type status: ``bool``
        :return: cropped PIL images.
        :rtype: ``list``
        """
        if self._verbose and self._print_update:
            print("\n\nComputing Crop Locations...")

        if data_frame is None:
            df = self.image_dataframe
        else:
            df = data_frame

        # Zip the relevant columns (faster than looping through the dataframe directly).
        to_predict = zip(*[df[i] for i in ('cached_images_path', 'lower_crop', 'upper_crop', 'vborder')])

        all_cropped_images = list()
        for cached_images_path, lower_crop, upper_crop, vborder in self._apply_status(to_predict, status):
            cropped_image = self._apply_cropping(cached_images_path,
                                                 lower_crop,
                                                 upper_crop,
                                                 vborder,
                                                 return_as_array,
                                                 convert_to_rgb)

            all_cropped_images.append(cropped_image)

        return all_cropped_images

    def _image_problems_predictions(self, status):
        """

        Carries out the actual computations for the ``visual_image_problems()`` method.

        :param status: display status bar. Defaults to True.
        :type status: ``bool``
        """
        # Apply crop
        cropped_images_for_analysis = self._cropper(data_frame=None, return_as_array=True)

        # Transform the cropped images into a form `ImageRecognitionCNN.predict()` can accept
        if self._verbose and self._print_update:
            print("\n\nPreparing Images for Neural Network...")
            verbose_prediction = True
        else:
            verbose_prediction= False

        transformed_images = load_and_scale_imgs(cropped_images_for_analysis, self._ircnn.img_shape, status=status)

        # Make the predictions and Save
        self.image_dataframe['visual_image_problems'] = self._ircnn.predict(list_of_images=[transformed_images],
                                                                            status=status,
                                                                            verbose=verbose_prediction)

    def visual_image_problems(self, new_analysis=False, status=True):
        """

        This method is powered by a Convolutional Neural Network which
        computes probabilities for the presence of problematic image properties or types.

        Currently, the model can identify the follow problems:

        - arrows in images
        - images arrayed as grids

        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``


        :Examples:

        >>> DataFrame['visual_image_problems']
        ...
        0 [('valid_img', 0.97158539), ('arrows', 0.066939682), ('grids', 0.0010551035)]
        1 [('valid_img', 0.98873705), ('arrows', 0.024444019), ('grids', 0.0001462775)]
        2 [('valid_img', 0.89019465), ('arrows', 0.16754828), ('grids', 0.009004808)]
        3 [('grids', 0.85855108), ('valid_img', 0.0002961561), ('arrows', 6.8026602e-09)]


        The first value in the tuple represents the problem identified and second
        value represents its associated probability. For example, in the final row
        we can see that the model strongly 'believes' both that the image is, in fact,
        an image composed of several smaller images (forming an image 'grid'). Conversely,
        it believes all of the other images are likely devoid of problems it has been
        trained to detect.
        """
        # Note: This method is a wrapper for `_image_problems_predictions()`
        if self._verbose and self._print_update:
            print("\n\nAnalyzing Images for Problems...")

        if 'visual_image_problems' not in self.image_dataframe.columns or new_analysis:
            self._image_problems_predictions(status=status)

    def auto_analysis(self, new_analysis=False, status=True):
        """

        Automatically use the class methods to analyze the ``image_dataframe`` using default
        parameter values for class methods.

        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        """
        # Permit Verbosity, if requested by the user upon class instantiation.
        self._print_update = True

        # Run Analysis Battery with Default Parameter Values
        self.grayscale_analysis(new_analysis=new_analysis)
        self.logo_analysis(new_analysis=new_analysis)
        self.border_analysis(new_analysis=new_analysis)

        # Compute Crop location
        self.crop_decision(new_analysis=new_analysis)

        # Generate predictions
        self.visual_image_problems(new_analysis=new_analysis, status=status)

        # Ban Verbosity
        self._print_update = False

    def auto_decision(self, img_problem_threshold, require_grayscale, valid_floor=0.01):
        """

        Automatically generate 'valid_image' column in the `image_dataframe`
        column by deciding whether or not images are valid using default parameter values for class methods.

        :param img_problem_threshold: a scalar from 0 to 1 which specifies the threshold value required
                                      to cause the image to be marked as invalid.
                                      For instance, a threshold value of `0.5` would mean that any image
                                      which contains a image problem probability above `0.5` will be marked
                                      as invalid.
        :type img_problem_threshold: ``float``
        :param require_grayscale: if True, require that images are grayscale to be considered valid.
        :type require_grayscale: ``bool``
        :param valid_floor: the smallest value needed for a 'valid_img' to be considered valid. Defaults to `0.01`.
        :type valid_floor: ``float``
        """
        # ToDo: make `require_grayscale` flexible s.t. it can be imposed only on images of a certain type (e.g., MRI).
        for i in ('grayscale', 'visual_image_problems'):
            if i not in self.image_dataframe.columns:
                raise KeyError("`image_dataframe` does not contain a {0} column.".format(i))

        def img_validity(x):
            if not items_null(x['grayscale']) and x['grayscale'] == False and require_grayscale:
                return False
            # Block if the 'image_caption' column suggests the presence of a problem.
            elif isinstance(x['image_problems_from_text'], (list, tuple)) and len(x['image_problems_from_text']):
                return False
            else:  # ToDo: add variable img_problem_threshold (e.g., arrows and grids).
                # if all image problem confidence is < img_problem_threshold, return True; else False.
                i = x['visual_image_problems']
                if i[0][0] == 'valid_img' and i[0][1] < valid_floor:
                    return False
                elif i[0][0] != 'valid_img' and i[0][1] > img_problem_threshold:
                    return False
                elif i[0][0] == 'valid_img' and i[1][1] > img_problem_threshold:
                    return False
                else:
                    return True

        self.image_dataframe['valid_image'] = self.image_dataframe.apply(img_validity, axis=1)

    def auto(self,
             img_problem_threshold=0.275,
             valid_floor=0.01,
             require_grayscale=True,
             new_analysis=False,
             status=True):
        """

        Automatically carry out all aspects of image preprocessing (recommended).

        :param img_problem_threshold: see ``auto_decision()``. Defaults to `0.275`.
        :type img_problem_threshold: ``float``
        :param valid_floor: the smallest value needed for a 'valid_img' to be considered valid. Defaults to `0.01`.
        :type valid_floor: ``float``
        :param require_grayscale: see ``auto_decision()``. Defaults to ``True``
        :type require_grayscale: ``bool``
        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        :return: the `image_dataframe`, complete with the results of all possible analyses
                (using default parameter values).
        :rtype: ``Pandas DataFrame``
        """
        # Run Auto Analysis
        self.auto_analysis(new_analysis=new_analysis, status=status)

        # Run Auto Decision
        self.auto_decision(img_problem_threshold=img_problem_threshold,
                           require_grayscale=require_grayscale,
                           valid_floor=valid_floor)

        return self.image_dataframe

    def save(self, save_rule, crop_images=True, convert_to_rgb=False, status=True):
        """

        Save processed images to disk.

        :param save_rule: the directory to save the images.
        :type save_rule: ``str``
        :param crop_images: Crop the images using analyses results from `border_analysis()` and
                            ``logo_analysis()``. Defaults to ``True``.
        :type crop_images: ``bool``
        :param convert_to_rgb: if ``True``, use the PIL library to convert the images to RGB. Defaults to ``False``.
        :type convert_to_rgb: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        """
        if 'valid_image' not in self.image_dataframe.columns:
            raise KeyError("`image_dataframe` must contain a 'valid_image' column which uses booleans to\n"
                           "indicate whether or not to include an entry in the cleaned dataset.\n"
                           "To automate this process, consider using the `auto_decision()` method.")

        # Limit this operation to subsection of the dataframe where valid_image is `True`.
        valid_df = self.image_dataframe[self.image_dataframe['valid_image'] == True].reset_index(drop=True).copy(deep=True)

        if crop_images:
            if self._verbose:
                print("\n\nCropping Images...")
            valid_df['image_to_return'] = self._cropper(data_frame=valid_df,
                                                        return_as_array=False,
                                                        convert_to_rgb=convert_to_rgb,
                                                        status=status)
        else:
            if self._verbose:
                print("\n\nLoading Images...")
            valid_df['image_to_return'] = self._pil_load(valid_df['cached_images_path'], convert_to_rgb, status)

        if self._verbose:
            print("\n\nSaving Images...")

        img_record = set()
        for index, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
            full_save_path = os.path.join(save_rule, row['cached_images_path'].split(os.sep)[-1])
            row['image_to_return'].save(full_save_path)
            img_record.add(full_save_path)

























