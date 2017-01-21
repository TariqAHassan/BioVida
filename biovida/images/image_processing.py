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
from PIL import ImageStat
from scipy.misc import imread

# General tools
from biovida.support_tools.support_tools import items_null

# Tools form the image subpackage
from biovida.images.image_tools import load_and_scale_imgs

# Models
from biovida.images.models.border_detection import border_detection
from biovida.images.models.img_classification import ImageRecognitionCNN
from biovida.images.models.template_matching import robust_match_template

# Suppress Pandas' SettingWithCopyWarning
pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------------------------
# General procedure (for Ultrasound, X-ray, CT and MRI):
# ---------------------------------------------------------------------------------------------
#
#   1. Check if grayscale                                                       X
#         - mark finding in dataframe.
#   2. Look for MedLine(R) logo                                                 P
#         - if true, try to crop
#   3. Look for text bar                                                        P
#         - if true, try crop
#   4. Look for border                                                          P
#         - if true, try to crop
#   5. Look for arrows or boxes *
#         - if true, ban
#   6. Look for image grids *
#         - if true, ban
#   7. Look for graphs *
#         - if true, ban
#   8. Look for faces *
#         - if true, ban
#   9. Look for other text in the image *
#         - if true, ban (or find crop).
#
# Legend:
#     * = requires machine learning
#     p = partially solved
#     X = solved
# ---------------------------------------------------------------------------------------------


def _extract_search_class_db(database_to_extract, search_class):
    """

    Extracts a database from the `search_class` parameter of ImageProcessing().

    :param database_to_extract: see ``ImageProcessing()``.
    :type database_to_extract: ``str``
    :param search_class: see ``ImageProcessing()``.
    :type search_class: ``OpenInterface Class``
    :return: extract database
    :rtype: ``Pandas DataFrame``
    """
    if database_to_extract == 'search':
        extracted_db = search_class.current_search_database
    elif database_to_extract == 'record':
        extracted_db = search_class.image_record_database
    else:
        raise ValueError("`database_to_extract` must be one of: 'search', 'record'.")

    if extracted_db is None:
        raise AttributeError("The {0} database provided was `None`.".format(database_to_extract))
    else:
        return extracted_db


class ImageProcessing(object):
    """

    This class is designed to allow easy analysis of cached image data.

    :param search_class: aninstance of the ``biovida.images.openi_interface.OpenInterface()`` class.
    :type search_class: ``OpenInterface Class``
    :param model_location: the location of the model for Convnet.
                          If `None`, the default model will be used. Defaults to ``None``.
    :type model_location: ``str``
    :param database_to_extract: 'search' to extract the ``current_search_database`` from ``search_class`` or
                                 'record' to extract ``image_record_database``. Defaults to 'search'.
    :type database_to_extract: ``str``
    :param verbose: if ``True``, print additional details. Defaults to ``False``.
    :type verbose: ``bool``

    :var image_dataframe: this is the search dataframe that was passed when instantiating the class and
                          contains a record of all analyses run as new columns.
    """
    def __init__(self, search_class, model_location=None, database_to_extract='search', verbose=True):
        if "OpenInterface" not in str(type(search_class)):
            raise ValueError("`search_class` must be a OpenInterface instance.")

        # Extract the search/record database
        self.image_dataframe = _extract_search_class_db(database_to_extract, search_class)

        # Extract path to the MedPix Logo
        self._medpix_path = os.path.join(search_class.root_path, "medpix_logo.png")

        # Spin up tqdm
        tqdm.pandas("status")

        self._verbose = verbose

        # Load the CNN
        self._ircnn = ImageRecognitionCNN()

        # Load the model weights and architecture.
        MODEL_PATH = pkg_resources.resource_filename('biovida', 'images/_resources/img_problems_model.h5')
        if model_location is None:
            self._ircnn.load(MODEL_PATH, override_existing=True)
        elif not isinstance(model_location, str):
            raise ValueError("`model_location` must either be a string or `None`.")
        elif os.path.isfile(model_location):
            self._ircnn.load(MODEL_PATH, override_existing=True)
        else:
            raise FileNotFoundError("'{0}' could not be located.".format(str(model_location)))

        # Load the visual image problems the model can detect
        self.model_classes = list(self._ircnn.data_classes.keys())

        # Container for images represented as `ndarrays`
        self._ndarrays_images = None

        # Switch to control verbosity
        self._print_update = False

    def _apply_status(self, x, status):
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
            self._ndarrays_images = [imread(i, flatten=True) for i in self.image_dataframe['img_cache_path']]

        if zip_with_column is not None:
            return zip(*[self._ndarrays_images, self.image_dataframe[zip_with_column]])
        else:
            return self._ndarrays_images

    def _grayscale_img(self, img_path):
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
        img = Image.open(img_path) # ToDo: find way to call only once inside this class (similar to _ndarray_extract)
        stat = ImageStat.Stat(img.convert("RGB"))
        return np.mean(stat.sum) == stat.sum[0]

    def grayscale_analysis(self, new_analysis=False):
        """

        Analyze the images to determine whether or not they are grayscale
        (uses the PIL image library).

        Note:
              - this tool is very conservative (very small amounts of 'color' will yield `False`).
              - the exception to the above rule is the *very rare* of an image which even split
                of red, green and blue.

        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        """
        if self._verbose and self._print_update:
            print("\n\nStarting Grayscale Analysis...")

        if 'grayscale' not in self.image_dataframe.columns or new_analysis:
            self.image_dataframe['grayscale'] = self.image_dataframe['img_cache_path'].progress_map(
                self._grayscale_img, na_action='ignore')

    def _logo_analysis_out(self, match, base_img_shape, output_params):
        """

        Decides the output for the `logo_analysis` function.
        If the bonding box is in an improbable location, NaN is returned.
        Otherwise, the bonding box, or some portion of it (i.e., the lower left) will be returned.

        :param match: the bounding box for the location which provided the highest quality match.
        :type match: ``dict``
        :param base_img_shape: the shape of the base image (where the algorithm was trying to find the given pattern).
                               Form: (x, y).
        :type base_img_shape: ``tuple``
        :param output_params: tuple of the form:
                                (threshold, xy_position_threshold[0], xy_position_threshold[1], return_full)
        :type output_params: ``tuple``
        :return: the output requested by the ``logo_analysis()`` method.
        :rtype: ``NaN``, ``dict`` or ``tuple``
        """
        match_quality_threshold, x_greater_check, y_greater_check, return_full = output_params

        # Check match quality
        if not isinstance(match, dict) or match['match_quality'] < match_quality_threshold:
            return np.NaN

        # Check the box is in the top right
        if match['box']['bottom_left'][0] < (base_img_shape[0] * x_greater_check) or \
                match['box']['bottom_left'][1] > (base_img_shape[1] * y_greater_check):
            return np.NaN

        if return_full:
            return match['box']
        else:
            return match['box']['bottom_left']

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
                match, base_img_shape = robust_match_template_wrapper(img)
                current = self._logo_analysis_out(match, base_img_shape, output_params)
                results.append(current)

        return results

    def logo_analysis(self,
                      match_quality_threshold=0.25,
                      xy_position_threshold=(1/3.0, 1/2.5),
                      base_top_cropping=0.14,
                      prop_scale=0.075,
                      scaling_lower_limit=0.25,
                      end_search_threshold=0.875,
                      base_resizes=(1.25, 2.75, 0.25),
                      return_full=False,
                      new_analysis=False,
                      status=True):
        """

        Search for the MexPix Logo. If located, with match quality above match_quality_threshold,
        populate the corresponding row of the 'medpix_logo_lower_left' column in the 'image_dataframe'
        with its a. full bonding box (if ``return_full`` is `True`) or lower left corner otherwise.

        :param match_quality_threshold: the minimum match quality required to accept the match.
                                        See: ``skimage.feature.match_template()`` for more.
        :type match_quality_threshold: ``float``
        :param xy_position_threshold: tuple of the form: (x_greater_check, y_greater_check).
                                      For instance the default (``(1/3.0, 1/2.5)``) requires that the
                                      x position of the logo is greater than 1/3 of the image's width
                                      and less than 1/2.5 of the image's height.
        :type xy_position_threshold: ``tuple``
        :param base_top_cropping: See: ``biovida.images.models.template_matching.robust_match_template()``.
        :type base_top_cropping: ``float``
        :param prop_scale: See: ``biovida.images.models.template_matching.robust_match_template()``.
        :type prop_scale: ``float``
        :param scaling_lower_limit: See: ``biovida.images.models.template_matching.robust_match_template()``.
        :type scaling_lower_limit: ``int`` or ``float``
        :param end_search_threshold: See: ``biovida.images.models.template_matching.robust_match_template()``.
        :type end_search_threshold:
        :param base_resizes: See: ``biovida.images.models.template_matching.robust_match_template()``.
        :type base_resizes: ``tuple``
        :param return_full: if ``True``, return a dictionary with the location of all four corners for the
                            logo's bounding box. Otherwise, only the bottom left corner will be returned.
                            Defaults to ``False``.
                            Note: ``True`` **cannot** be used in conjunction with 'auto' methods.
        :type return_full: ``bool``
        :param new_analysis: rerun the analysis if it has already been computed.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to True.
        :type status: ``bool``
        :return: the bottom left corner of the bounding box for the medpix logo.
        :rtype: ``tuple``
        """
        # Note: wraps ``biovida.images.models.template_matching.robust_match_template()``.

        if 'medpix_logo_lower_left' in self.image_dataframe.columns and not new_analysis:
            return None

        if self._verbose and self._print_update:
            print("\n\nStarting Logo Analysis...")

        # Package Params
        output_params = (match_quality_threshold, xy_position_threshold[0], xy_position_threshold[1], return_full)

        # Load the Pattern. ToDo: Allow for non MedPix logos logos.
        medline_template_img = imread(self._medpix_path, flatten=True)

        def robust_match_template_wrapper(img):
            return robust_match_template(pattern_img=medline_template_img,
                                         base_img=img,
                                         base_top_cropping=base_top_cropping,
                                         prop_scale=prop_scale,
                                         scaling_lower_limit=scaling_lower_limit,
                                         end_search_threshold=end_search_threshold,
                                         base_resizes=base_resizes)

        # Run the algorithm searching for the medpix logo in the base image
        results = self._logo_processor(robust_match_template_wrapper, output_params, status)

        # Update dataframe with the (x, y) values of the lower left corner of the logo's bonding box.
        self.image_dataframe['medpix_logo_lower_left'] = results

    def border_analysis(self
                        , signal_strength_threshold=0.25
                        , min_border_separation=0.15
                        , lower_bar_search_space=0.9
                        , new_analysis=False
                        , status=True):
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

    def _h_crop_top_decision(self, x):
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

    def _h_crop_lower_decision(self, x):
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

    def _apply_cropping(self, img_cache_path, lower_crop, upper_crop, vborder, return_as_array=True, convert_to_rgb=True):
        """

        Applies cropping to a specific image.

        :param img_cache_path: path to the image
        :type img_cache_path: ``str``
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
        converted_image = Image.open(img_cache_path)

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

    def _cropper(self, data_frame=None, return_as_array=True, include_path=False, convert_to_rgb=True, status=True):
        """

        Uses `_apply_cropping()` to apply cropping to images in a dataframe.

        :param data_frame: a dataframe with 'img_cache_path', 'lower_crop', 'upper_crop' and 'vborder' columns.
                          if None ``image_dataframe`` is used.
        :type data_frame: ``None`` or ``Pandas DataFrame``
        :param return_as_array: if True, convert the PIL object to an ``ndarray``. Defaults to True.
        :type return_as_array: ``bool``
        :param include_path: if ``True``, generate a list of lists of the form: ``[(PIL image, path to the image)...]``.
                             if ``False``, generate a list of lists of the form ``[PIL image, PIL Image, PIL Image...]``.
        :type include_path: ``bool``
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
        to_predict = zip(*[df[i] for i in ('img_cache_path', 'lower_crop', 'upper_crop', 'vborder')])

        all_cropped_images = list()
        for img_cache_path, lower_crop, upper_crop, vborder in self._apply_status(to_predict, status):
            cropped_image = self._apply_cropping(img_cache_path,
                                                 lower_crop,
                                                 upper_crop,
                                                 vborder,
                                                 return_as_array,
                                                 convert_to_rgb)

            if include_path:
                all_cropped_images.append((cropped_image, img_cache_path))
            else:
                all_cropped_images.append(cropped_image)

        return all_cropped_images

    def _image_problems_predictions(self, status):
        """

        Carries out the actual computations for the ``img_problems()`` method.

        :param status: display status bar. Defaults to True.
        :type status: ``bool``
        """
        # Apply crop
        cropped_images_for_analysis = self._cropper(data_frame=None, return_as_array=True)

        # Transform the cropped images into a form `ImageRecognitionCNN.predict()` can accept
        if self._verbose and self._print_update:
            print("\n\nPreparing Images for Neural Network...")
        transformed_images = load_and_scale_imgs(cropped_images_for_analysis, self._ircnn.img_shape, status=status)

        # Make the predictions and Save
        self.image_dataframe['img_problems'] = self._ircnn.predict([transformed_images], status=status)

    def img_problems(self, new_analysis=False, status=True):
        """

        This method is powered by a Convolutional Neural Network which
        computes probabilities for the presence of problematic image properties or types.

        Currently, the model can identify the follow problems:

            - arrows in images
            - ellipses in images
            - images arrayed as grids

        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``


        :Examples:

        >>> DataFrame['img_problems']
        ...
        0   [(arrows, 0.197112), (grids, 0.0109332)]
        1   [(arrows, 0.211948), (grids, 0.00918275)]
        2   [(arrows, 0.473652), (grids, 0.00578115)]
        3   [(arrows, 0.43337),  (grids, 0.00857231)]
        4   [(grids, 0.928362), (arrows, 1.10526e-06)]

        The first value in the tuple represents the problem identified and second
        value represents its associated probability (for the sake of *simplicity*,
        this be interpreted as the model's confidence).

        For example, in the final row we can see that the model strongly 'believes' both
        that the image is, in fact, a grid of images and that it does not contain arrows.
        """
        # Note: This method is a wrapper for `_image_problems_predictions()`
        if self._verbose and self._print_update:
            print("\n\nAnalyzing Images for Problems...")

        if 'img_problems' not in self.image_dataframe.columns or new_analysis:
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
        self.img_problems(new_analysis=new_analysis, status=status)

        # Ban Verbosity
        self._print_update = False

    def auto_decision(self, img_problem_threshold, require_grayscale):
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
        """
        # ToDo: make `require_grayscale` flexible s.t. it can be imposed only on images of a certain type (e.g., MRI).
        for i in ('grayscale', 'img_problems'):
            if i not in self.image_dataframe.columns:
                raise KeyError("`image_dataframe` does not contain a {0} column.".format(i))

        def img_validity(x):
            if not items_null(x['grayscale']) and x['grayscale'] == False and require_grayscale:
                    return False
            else:  # ToDo: add variable img_problem_threshold (e.g., arrows and grids).
                # if all image problem confidence is < img_problem_threshold, return True; else False.
                return all(x[1] < img_problem_threshold for x in x['img_problems'])

        self.image_dataframe['valid_image'] = self.image_dataframe.apply(img_validity, axis=1)

    def auto(self, img_problem_threshold=0.45, require_grayscale=True, new_analysis=False, status=True):
        """

        Automatically carry out all aspects of image preprocessing (recommended).

        :param img_problem_threshold: see `auto_decision()`. Defaults to 0.45.
        :type img_problem_threshold: ``float``
        :param require_grayscale: see `auto_decision()`. Defaults to ``True``
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
        self.auto_decision(img_problem_threshold=img_problem_threshold, require_grayscale=require_grayscale)

        return self.image_dataframe

    def save(self, save_path, crop_images=True, convert_to_rgb=False, block_duplicates=True, status=True):
        """

        Save processed images to disk.

        :param save_path: the directory to save the images.
        :type save_path: ``str``
        :param crop_images: Crop the images using analyses results from `border_analysis()` and
                            `logo_analysis()`. Defaults to ``True``.
        :type crop_images: ``bool``
        :param convert_to_rgb: if ``True``, use the PIL library to convert the images to RGB. Defaults to ``False``.
        :type convert_to_rgb: ``bool``
        :param block_duplicates: if ``True``, prohibit writing duplicates. Defaults to ``True``.
        :type block_duplicates: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        """
        if 'valid_image' not in self.image_dataframe.columns:
            raise KeyError("`image_dataframe` must contain a 'valid_image' column which uses booleans to\n"
                           "indicate whether or not to include an entry in the cleaned dataset.\n"
                           "To automate this process, consider using the `auto_decision()` method.")

        # Limit this operation to subsection of the dataframe where valid_image is `True`.
        valid_df = self.image_dataframe[self.image_dataframe['valid_image'] == True].reset_index(drop=True)

        if crop_images:
            if self._verbose and self._print_update:
                print("\n\nCropping Images...")
            images_to_return = self._cropper(valid_df,
                                             return_as_array=False,
                                             include_path=True,
                                             convert_to_rgb=convert_to_rgb,
                                             status=status)
        else:
            images_to_return = self._pil_load(valid_df['img_cache_path'], convert_to_rgb, status)

        # Save to disk
        img_record = set()
        for img, path in self._apply_status(images_to_return, status):
            full_save_path = os.path.join(save_path, path.split(os.sep)[-1])

            if block_duplicates:
                if full_save_path not in img_record:
                    img_record.add(full_save_path)
                    img.save(full_save_path)
            else:
                img.save(full_save_path)







































