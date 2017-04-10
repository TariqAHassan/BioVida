# coding: utf-8

"""

    Image Processing
    ~~~~~~~~~~~~~~~~

"""
import os
import json
import requests
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageStat
from scipy.misc import imread
from collections import defaultdict
from os.path import join as os_join
from six.moves.urllib.parse import urljoin
from os.path import basename as os_basename

# General tools
from biovida.support_tools.support_tools import (tqdm,
                                                 items_null,
                                                 data_frame_col_drop,
                                                 list_to_bulletpoints)

# Tools form the image subpackage
from biovida.images._image_tools import load_and_scale_images

# Open-i Support Tools
from biovida.images._interface_support.openi.openi_support_tools import (nonessential_openi_columns,
                                                                         grayscale_openi_modalities)

# Models
from biovida.images.models.border_detection import border_detection
from biovida.images.models.template_matching import robust_match_template
from biovida.images.models.image_classification import ImageClassificationCNN

# Suppress Pandas' SettingWithCopyWarning
pd.options.mode.chained_assignment = None


class OpeniImageProcessing(object):
    """

    This class is designed to allow easy analysis and cleaning of cached Open-i image data.

    :param instance: an instance of the ``biovida.images.openi_interface.OpenInterface()`` class.
    :type instance: ``OpenInterface Class``
    :param db_to_extract: ``records_db`` or``cache_records_db``. Defaults to 'records_db'.
    :type db_to_extract: ``str``
    :param model_location: the location of the model for Convnet.
                           If `None`, the default model will be used. Defaults to ``None``.
    :type model_location: ``str``
    :param download_override: If ``True``, download a new copy of the 'visual_image_problems_model' weights (and
                              associated resources) regardless of whether or not files with these names are
                              already cached. Defaults to ``False``.
    :type download_override: ``bool``
    :param verbose: if ``True``, print additional details. Defaults to ``False``.
    :type verbose: ``bool``

    :var image_dataframe: this is the dataframe that was passed when instantiating the class and
                          contains a cache of all analyses run as new columns.
    """

    def _obtain_model_resources(self, download_override):
        """
        
        Obtain the 'visual_image_problems_model' 
        
        :param download_override: If ``True``, download the 'visual_image_problems_model' weights (and
                                  associated resources) regardless of whether or not these files are already cached.
        :type download_override: ``bool``
        """
        resources_path = os_join(self.instance._created_image_dirs['aux'], 'image_processing_resources')
        if not os.path.isdir(resources_path):
            os.makedirs(resources_path)

        base_url = 'https://raw.githubusercontent.com/TariqAHassan/BioVida/master/data/model_resources/'

        required_resources = ["trained_open_i_modality_types.json",
                              "visual_image_problems_model.h5",
                              "visual_image_problems_model_support.p"]

        def download(url, file_path):
            response = requests.get(url)
            with open(file_path, 'wb') as file:
                file.write(response.content)

        for resource in required_resources:
            file_path = os_join(resources_path, resource)
            if not os.path.isfile(file_path) or download_override:
                if self._verbose:
                    print("Downloading '{0}'... ".format(resource))
                download(url=urljoin(base_url, resource), file_path=file_path)

        self._model_path = os_join(resources_path, "visual_image_problems_model.h5")

        # Load 'trained_open_i_modality_types.json' into memory
        with open(os_join(resources_path, 'trained_open_i_modality_types.json')) as json_data:
            self.trained_open_i_modality_types = json.load(json_data)

    @staticmethod
    def _extract_db(instance, db_to_extract):
        """

        Extracts a database from the `instance` parameter of OpeniImageProcessing().

        :param db_to_extract: see ``OpeniImageProcessing()``.
        :type db_to_extract: ``str``
        :param instance: see ``OpeniImageProcessing()``.
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
                            "type DataFrame, not: '{1}'.".format(db_to_extract, type(extract).__name__))

    def __init__(self,
                 instance,
                 db_to_extract='records_db',
                 model_location=None,
                 download_override=False,
                 verbose=True):
        self._verbose = verbose
        self.db_to_extract = db_to_extract
        self._cache_path = getattr(instance, '_cache_path')
        self.known_image_problems = ('arrows', 'asterisks', 'grids')

        if "OpeniInterface" != type(instance).__name__:
            raise ValueError("`instance` must be a `OpeniInterface` instance.")
        self.instance = instance

        # Extract the records_db/cache_records_db database
        self.image_dataframe = self._extract_db(instance, db_to_extract)
        self.image_dataframe_cleaned = None

        if 'cached_images_path' not in self.image_dataframe.columns:
            raise KeyError("No 'cached_images_path' column in '{0}'.".format(db_to_extract))

        # Extract path to the MedPix Logo
        self._medpix_path = instance._created_image_dirs['medpix_logo']

        # Load the CNN
        self._ircnn = ImageClassificationCNN()
        self.trained_open_i_modality_types = None
        self._model_path = None

        self._obtain_model_resources(download_override=download_override)

        # Load the model weights and architecture.
        if model_location is None:
            self._ircnn.load(self._model_path, override_existing=True)
        elif not isinstance(model_location, str):
            raise ValueError("`model_location` must either be a string or `None`.")
        elif os.path.isfile(model_location):
            self._ircnn.load(self._model_path, override_existing=True)
        else:
            raise FileNotFoundError("'{0}' could not be located.".format(str(model_location)))

        # Load the visual image problems the model can detect
        self.model_classes = list(self._ircnn.data_classes.keys())

        # Container for images represented as `ndarrays`
        self._ndarrays_images = None

    @property
    def image_dataframe_short(self):
        """Return `image_dataframe` with nonessential columns removed."""
        return data_frame_col_drop(self.image_dataframe, nonessential_openi_columns, 'image_dataframe')

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
        def conversion(image):
            return (Image.open(image).convert('RGB'), image) if convert_to_rgb else (Image.open(image), image)
        return [conversion(i) for i in tqdm(image_paths, desc="Loading Images", disable=not status)]

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
            return list(zip(*[self._ndarrays_images, self.image_dataframe[zip_with_column]]))
        else:
            return self._ndarrays_images

    @staticmethod
    def _grayscale_image(image_path):
        """

        Computes whether or not an image is grayscale.
        See the `grayscale_analysis()` method for caveats.

        :param image_path: path to an image.
        :type image_path:``str``
        :return: ``True`` if grayscale, else ``False``.
        :rtype: ``bool``
        """
        # See: http://stackoverflow.com/q/23660929/4898004
        if image_path is None or items_null(image_path):
            return np.NaN
        image = Image.open(image_path)  # ToDo: find way to call only once inside this class (similar to _ndarray_extract())
        stat = ImageStat.Stat(image.convert("RGB"))
        return np.mean(stat.sum) == stat.sum[0]

    def grayscale_analysis(self, new_analysis=False, status=True):
        """

        Analyze the images to determine whether or not they are grayscale
        (uses the PIL image library).

        Note:
              - this tool is very conservative (very small amounts of 'color' will yield `False`).
              - the exception to the above rule is the *very rare* case of an image which even split
                of red, green and blue.

        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        """
        if 'grayscale' not in self.image_dataframe.columns or new_analysis:
            column = self.image_dataframe['cached_images_path']
            grayscale = [self._grayscale_image(i) for i in tqdm(column, desc='Grayscale Analysis', disable=not status)]
            self.image_dataframe['grayscale'] = grayscale

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
        match_quality_threshold, x_greater_check, y_greater_check = output_params

        # Unpack ``analysis_results``
        bounding_box = analysis_results['bounding_box']
        match_quality = analysis_results['match_quality']
        base_image_shape = analysis_results['base_image_shape']

        # Check match quality.
        if bounding_box is None or match_quality < match_quality_threshold:
            return np.NaN

        # Check the box is in the top right (as defined by ``x_greater_check`` and ``y_greater_check``).
        if bounding_box['bottom_left'][0] < (base_image_shape[0] * x_greater_check) or \
                        bounding_box['bottom_left'][1] > (base_image_shape[1] * y_greater_check):
            return np.NaN

        return bounding_box

    def _logo_processor(self, robust_match_template_wrapper, output_params, status):
        """

        Performs the actual analysis for ``logo_analysis()``, searching for the
        MedPix logo in the images.

        :param robust_match_template_wrapper: wrapper generated inside of ``logo_analysis()``
        :type robust_match_template_wrapper: ``func``
        :param output_params: tuple of the form:

                        ``(match_quality_threshold, xy_position_threshold[0], xy_position_threshold[1])``

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

        for image, journal in tqdm(to_analyze, desc='Logo Analysis', disable=not status):
            if 'medpix' not in str(journal).lower():
                results.append(np.NaN)
            else:
                analysis_results = robust_match_template_wrapper(image)
                current = self._logo_analysis_out(analysis_results, output_params)
                results.append(current)

        return results

    def logo_analysis(self,
                      match_quality_threshold=0.25,
                      xy_position_threshold=(1 / 3.0, 1 / 2.5),
                      base_resizes=(0.5, 2.5, 0.1),
                      end_search_threshold=0.875,
                      base_image_cropping=(0.15, 0.5),
                      new_analysis=False,
                      status=True):
        """

        Search for the MedPix Logo. If located, with match quality above match_quality_threshold,
        populate the the 'medpix_logo_bounding_box' of ``image_dataframe`` with its bounding box.

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
        :param base_image_cropping: See: ``biovida.images.models.template_matching.robust_match_template()``
        :type base_image_cropping: ``tuple``
        :param new_analysis: rerun the analysis if it has already been computed.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        """
        # Note: this method wraps ``biovida.images.models.template_matching.robust_match_template()``.
        if 'medpix_logo_bounding_box' in self.image_dataframe.columns and not new_analysis:
            return None

        # Package Params
        output_params = (match_quality_threshold, xy_position_threshold[0], xy_position_threshold[1])

        # Load the Pattern. ToDo: Allow for non MedPix logos logos.
        medpix_template_image = imread(self._medpix_path, flatten=True)

        def robust_match_template_wrapper(image):
            return robust_match_template(pattern_image=medpix_template_image,
                                         base_image=image,
                                         base_resizes=base_resizes,
                                         end_search_threshold=end_search_threshold,
                                         base_image_cropping=base_image_cropping)

        # Run the algorithm searching for the medpix logo in the base image
        self.image_dataframe['medpix_logo_bounding_box'] = self._logo_processor(robust_match_template_wrapper,
                                                                                output_params=output_params,
                                                                                status=status)

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

        def ba_func(image):
            return border_detection(image,
                                    signal_strength_threshold=signal_strength_threshold,
                                    min_border_separation=min_border_separation,
                                    lower_bar_search_space=lower_bar_search_space,
                                    report_signal_strength=False,
                                    rescale_input_ndarray=True)

        to_analyze = self._ndarray_extract()

        # Run the analysis
        border_analysis = [ba_func(i) for i in tqdm(to_analyze, desc='Border Analysis', disable=not status)]

        # Convert to a dataframe
        ba_df = pd.DataFrame(border_analysis).fillna(np.NaN)

        # Update datafame
        self.image_dataframe['hbar'] = ba_df['hbar']
        self.image_dataframe['hborder'] = ba_df['hborder']
        self.image_dataframe['vborder'] = ba_df['vborder']

    @staticmethod
    def _h_crop_top_decision(x):
        """

        Choose lowest horizontal cropping point.
        Solves: upper 'hborder' vs 'medpix_logo_bounding_box'
        (dictionary key = 'bottom_left').

        :param x: data passed through Pandas ``DataFrame.apply()`` method.
        :type x: ``Pandas Object``
        :return: the lowest crop location.
        :rtype: ``int`` or ``float``
        """
        # Note: hborder = [top, lower]; medpix_logo_bounding_box['bottom_left'] = [x, y].
        # That is, `hborder[0]` must be compared with `medpix_logo_bounding_box['bottom_left'][1]`.
        crop_candidates = list()
        if isinstance(x['hborder'], (list, tuple)):
            crop_candidates.append(x['hborder'][0])
        if isinstance(x['medpix_logo_bounding_box'], dict):
            lower_left = x['medpix_logo_bounding_box'].get('bottom_left', None)
            if isinstance(lower_left, (list, tuple)):
                crop_candidates.append(lower_left[1])

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
        
        Decide where to crop the images, if at all.

        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        """
        if all(x in self.image_dataframe.columns for x in ['upper_crop', 'lower_crop']) and not new_analysis:
            return None

        for i in ('medpix_logo_bounding_box', 'hborder', 'hbar'):
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
        :param vborder: yield of the ``border_analysis()`` method
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
            image_to_save = converted_image.convert("RGB")
        else:
            image_to_save = converted_image

        if return_as_array:
            return np.asarray(image_to_save)
        else:
            return image_to_save

    def _cropper(self, data_frame=None, return_as_array=True, convert_to_rgb=True, status=True):
        """

        Uses `_apply_cropping()` to apply cropping to images in a dataframe.

        :param data_frame: a dataframe with 'cached_images_path', 'lower_crop', 'upper_crop' and 'vborder' columns.
                          If ``None`` ``image_dataframe`` is used.
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
        if isinstance(data_frame, pd.DataFrame):
            df = data_frame
        else:
            df = self.image_dataframe

        all_cropped_images = list()
        for index, row in tqdm(df.iterrows(), desc='Cropping Images', disable=not status, total=len(df)):
            cropped_image = self._apply_cropping(cached_images_path=row['cached_images_path'],
                                                 lower_crop=row['lower_crop'],
                                                 upper_crop=row['upper_crop'],
                                                 vborder=row['vborder'],
                                                 return_as_array=return_as_array,
                                                 convert_to_rgb=convert_to_rgb)
            all_cropped_images.append(cropped_image)

        return all_cropped_images

    def visual_image_problems(self, limit_to_known_modalities=True, new_analysis=False, status=True):
        """

        This method is powered by a Convolutional Neural Network which
        computes probabilities for the presence of problematic image properties or types.

        Currently, the model can identify the follow problems:

        - arrows in images
        - images arrayed as grids

        :param limit_to_known_modalities: if ``True``, remove model predicts for image modalities
                                          the model has not explicitly been trained on. Defaults to ``True``.
        :type limit_to_known_modalities: ``bool``
        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``


        :Examples:

        >>> DataFrame['visual_image_problems']
        ...
        0 [('valid_image', 0.82292306), ('text', 0.13276383), ('arrows', 0.10139297), ('grids', 0.021935554)]
        1 [('valid_image', 0.76374823), ('arrows', 0.1085605), ('grids', 0.0024915827), ('text', 0.00037114936)]
        2 [('valid_image', 0.84319711), ('text', 0.10483728), ('arrows', 0.06458132), ('grids', 0.0125442)]
        3 [('valid_image', 0.84013706), ('arrows', 0.090836897), ('text', 0.055015128), ('grids', 0.0088913934)]

        The first value in the tuple represents the problem identified and second
        value represents its associated probability.
        """
        if 'visual_image_problems' in self.image_dataframe.columns and not new_analysis:
            return None

        cropped_images_for_analysis = self._cropper(return_as_array=True, status=status)

        transformed_images = load_and_scale_images(list_of_images=cropped_images_for_analysis,
                                                   image_size=self._ircnn.image_shape, status=status,
                                                   desc='Preparing Images')

        # Scan Images for Visual Problems with Neural Network
        self.image_dataframe['visual_image_problems'] = self._ircnn.predict(list_of_images=[transformed_images],
                                                                            status=status, desc='Scanning for Problems')

        if limit_to_known_modalities:  # ToDo: Temporary. Future: avoid passing through the model in the first place.
            for index, row in self.image_dataframe.iterrows():
                if row['image_modality_major'] not in self.trained_open_i_modality_types:
                    self.image_dataframe.set_value(index, 'image_modality_major', np.NaN)

    def auto_analysis(self, limit_to_known_modalities=True, new_analysis=False, status=True):
        """

        Automatically use the class methods to analyze the ``image_dataframe`` using default
        parameter values for class methods.

        :param limit_to_known_modalities: if ``True``, remove model predicts for image modalities
                                          the model has not explicitly been trained on. Defaults to ``True``.
        :type limit_to_known_modalities: ``bool``
        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        """
        # Run Analysis Battery with Default Parameter Values
        self.grayscale_analysis(new_analysis=new_analysis, status=status)
        self.logo_analysis(new_analysis=new_analysis, status=status)
        self.border_analysis(new_analysis=new_analysis, status=status)

        # Compute Crop location
        self.crop_decision(new_analysis=new_analysis)

        # Generate predictions
        self.visual_image_problems(limit_to_known_modalities=limit_to_known_modalities,
                                   new_analysis=new_analysis, status=status)

    @staticmethod
    def _invalid_image_tests(row, problems_to_ignore, valid_floor, image_problem_threshold=None):
        """

        Tests to determine if ``row`` references an image with properties and/or features
        likely to be problematic when training a model.

        :param row: as passed by ``pandas.DataFrame.apply(func, axis=1)``.
        :type row: ``Pandas Series``
        :param problems_to_ignore: see ``auto_decision()``.
        :type problems_to_ignore: ``None``, ``list`` or ``tuple``
        :param valid_floor: see ``auto_decision()``.
        :type valid_floor: ``float``
        :param image_problem_threshold: a scalar from 0 to 1 which specifies the threshold value required
                                        to cause the image to be marked as invalid.
                                        For instance, a threshold value of `0.5` would mean that any image
                                        which contains a image problem probability above `0.5` will be marked
                                        as invalid. 
                                        
                                        NOTE: Currently not in use.
                                        
        :type image_problem_threshold: ``float``
        :return: a list of the form ``[invalid image (boolean), reasons for decision if the former is True]``, wrapped
                 in a pandas series so it can be neatly split into two columns when called via. ``DataFrame.apply()``.
        :rtype: ``Pandas Series``
        """
        def image_problems_from_text_test(ipft):
            problem = False
            if not isinstance(problems_to_ignore, (list, tuple)) and \
                    isinstance(ipft, (list, tuple)) and len(ipft):
                problem = True
            elif (isinstance(problems_to_ignore, (list, tuple)) and
                  isinstance(ipft, (list, tuple)) and len([i for i in ipft if i not in problems_to_ignore])):
                problem = True
            return ['image_problems_from_text'] if problem else []

        def visual_image_problems_test(vip):
            if isinstance(problems_to_ignore, (list, tuple)):
                vip_ = [i for i in vip if i[0] not in problems_to_ignore]
            else:
                vip_ = vip

            problem = False
            if len(vip_) == 1:
                if vip_[0][0] == 'valid_image' and vip_[0][1] < valid_floor:
                    problem = True
            else:
                if vip_[0][0] != 'valid_image':
                    problem = True
                elif vip_[0][0] == 'valid_image' and vip_[0][1] < valid_floor:
                    problem = True
            return ['visual_image_problems'] if problem else []

        reasons = list()
        if row['grayscale'] is False and row['image_modality_major'] in grayscale_openi_modalities:
            reasons.append('grayscale')
        if isinstance(row['image_problems_from_text'], (list, tuple)):
            reasons += image_problems_from_text_test(ipft=row['image_problems_from_text'])
        if isinstance(row['visual_image_problems'], (list, tuple)):
            reasons += visual_image_problems_test(vip=row['visual_image_problems'])

        return pd.Series([len(reasons) > 0, tuple(sorted(reasons)) if len(reasons) else None])

    def auto_decision(self, valid_floor=0.8, problems_to_ignore=None):
        """

        Automatically generate 'invalid_image' column in the `image_dataframe`
        column by deciding whether or not images are valid using default parameter values for class methods.

        :param valid_floor: the smallest value needed for a 'valid_image' to be considered valid. Defaults to `0.8`.
        
                .. note::
                
                    For an image to be considered 'valid', the most likely categorization for the image must be
                    'valid_image' and the probability that the model assigns when placing it in this category
                    must be great than or equal to ``valid_floor``.
        
        :type valid_floor: ``float``
        :param problems_to_ignore: image problems to ignore. See ``INSTANCE.known_image_problems`` for valid values.
                                   Defaults to ``None``.
        :type problems_to_ignore: ``None``, ``list`` or ``tuple``
        """
        for i in ('grayscale', 'image_problems_from_text', 'visual_image_problems'):
            if i not in self.image_dataframe.columns:
                raise KeyError("`image_dataframe` does not contain a '{0}' column.".format(i))

        if isinstance(problems_to_ignore, (list, tuple)):
            for i in problems_to_ignore:
                if i not in self.known_image_problems or i == 'valid_image':
                    raise ValueError("`problems_to_ignore` may only contain the following:\n"
                                     "{0}".format(list_to_bulletpoints(self.known_image_problems)))
        elif problems_to_ignore is not None:
            raise ValueError("`problems_to_ignore` must be a `string`, `list` or `tuple`.")

        test_results = self.image_dataframe.apply(
            lambda r: self._invalid_image_tests(r, problems_to_ignore, valid_floor), axis=1)
        self.image_dataframe['invalid_image'] = test_results[0]
        self.image_dataframe['invalid_image_reasons'] = test_results[1].fillna(np.NaN)

    def auto(self, valid_floor=0.8, limit_to_known_modalities=True,
             problems_to_ignore=None, new_analysis=False, status=True):
        """

        Automatically carry out all aspects of image preprocessing (recommended).

        :param valid_floor: the smallest value needed for a 'valid_image' to be considered valid. Defaults to `0.8`.
        :type valid_floor: ``float``
        :param limit_to_known_modalities: if ``True``, remove model predicts for image modalities
                                          the model has not explicitly been trained on. Defaults to ``True``.
        :type limit_to_known_modalities: ``bool``
        :param problems_to_ignore: image problems to ignore. See ``INSTANCE.known_image_problems`` for valid values.
                                   Defaults to ``None``.
        :type problems_to_ignore: ``None``, ``list`` or ``tuple``
        :param new_analysis: rerun the analysis if it has already been computed. Defaults to ``False``.
        :type new_analysis: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        :return: the `image_dataframe`, complete with the results of all possible analyses
                (using default parameter values).
        :rtype: ``Pandas DataFrame``
        """
        # Run Auto Analysis
        self.auto_analysis(limit_to_known_modalities=limit_to_known_modalities,
                           new_analysis=new_analysis, status=status)

        # Run Auto Decision
        self.auto_decision(problems_to_ignore=problems_to_ignore,
                           valid_floor=valid_floor)

        return self.image_dataframe

    def _save_method_error_checking(self, output_rule, action):
        """

        Check for error that would cause ``save()`` to fail.

        :param output_rule: see ``output()``
        :type output_rule: ``str`` or ``func``
        :param action: see ``output``
        :type action: ``str``
        """
        if action not in ('copy', 'ndarray'):
            raise ValueError("`action` must be either 'copy' or 'ndarray'.")
        if not isinstance(output_rule, str) and not callable(output_rule):
            raise TypeError("`output_rule` must be a string or function.")
        if 'invalid_image' not in self.image_dataframe.columns:
            raise KeyError("`image_dataframe` must contain a 'invalid_image' column which uses booleans to\n"
                           "indicate whether or not to include an entry in the cleaned dataset.\n"
                           "To automate this process, consider using the `auto_decision()` method.")

    def clean_image_dataframe(self, crop_images=True, convert_to_rgb=False, status=True):
        """

        Define a dataframe with rows of images found to be 'valid'.
        These 'valid' images are cleaned and stored as PIL images in a ``'cleaned_image'`` column.
        
        .. note::
            
            The DataFrame this method creates can be viewed with ``INSTANCE.image_dataframe_cleaned``.

        :param crop_images: Crop the images using analyses results from `border_analysis()` and
                            ``logo_analysis()``. Defaults to ``True``.
        :type crop_images: ``bool``
        :param convert_to_rgb: if ``True``, use the PIL library to convert the images to RGB. Defaults to ``False``.
        :type convert_to_rgb: ``bool``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``
        :return: ``self.image_dataframe`` where the 'invalid_image' column is ``True``, with the addition of
                  a 'cleaned_image' populated by PIL image to be saved to disk.
        :rtype: ``Pandas DataFrame``
        """
        if 'invalid_image' not in self.image_dataframe:
            raise KeyError("No `invalid_image` column in `image_dataframe`,\n"
                           "which is required to determine output.\n"
                           "Consider calling the ``auto()`` method.")

        image_dataframe_cleaned = self.image_dataframe[
            self.image_dataframe['invalid_image'] != True].reset_index(drop=True).copy(deep=True)

        if crop_images:
            image_dataframe_cleaned['cleaned_image'] = self._cropper(data_frame=image_dataframe_cleaned,
                                                                     return_as_array=False,
                                                                     convert_to_rgb=convert_to_rgb,
                                                                     status=status)
        else:
            image_dataframe_cleaned['cleaned_image'] = self._pil_load(image_dataframe_cleaned['cached_images_path'],
                                                                      convert_to_rgb=convert_to_rgb, status=status)

        self.image_dataframe_cleaned = image_dataframe_cleaned

    def output(self,
               output_rule,
               create_dirs=False,
               allow_overwrite=True,
               action='copy',
               status=True,
               **kwargs):
        """

        Save processed images to disk.

        :param output_rule:

            - if a ``str``: the directory to save the images.
            - if a ``function``: it must (1) accept a single parameter (argument) and (2) return system path(s)
              [see example below].

        :type output_rule: ``str`` or ``func``
        :param create_dirs: if ``True``, create directories returned by ``divvy_rule`` if they do not exist.
                            Defaults to ``False``.
        :type create_dirs: ``bool``
        :param allow_overwrite: if ``True`` allow existing images to be overwritten. Defaults to ``True``.
        :type allow_overwrite: ``bool``
        :param action: one of 'copy', 'ndarray'.
        :type action: ``str``
        :param status: display status bar. Defaults to ``True``.
        :type status: ``bool``

        :Example:

        >>> from biovida.images import OpeniInterface
        >>> from biovida.images import OpeniImageProcessing
        ...
        >>> opi = OpeniInterface()
        >>> opi.search(image_type='mri')
        >>> opi.pull()
        ...
        >>> ip = OpeniImageProcessing(opi)
        >>> ip.auto()
        
        Next, prune invalid images 
        
        >>> ip.clean_image_dataframe()

        A Simple Output Rule

        >>> ip.output('/your/path/here/images')

        A More Complex Output Rule

        >>> def my_save_rule(row):
        >>>     if isinstance(row['abstract'], str) and 'lung' in row['abstract']:
        >>>         return '/your/path/here/lung_images'
        >>>     elif isinstance(row['abstract'], str) and 'heart' in row['abstract']:
        >>>         return '/your/path/here/heart_images'
        ...
        >>> ip.save(my_save_rule)

        """
        # Limit to 'valid' images
        if not isinstance(self.image_dataframe_cleaned, pd.DataFrame):
            raise TypeError("`image_dataframe_cleaned` is not a DataFrame.\n"
                            "The `clean_image_dataframe()` method must be called before ``output()``.")

        self._save_method_error_checking(output_rule=output_rule, action=action)
        allow_write = kwargs.get('allow_write', True)
        ndarray_with_path = kwargs.get('ndarray_with_path', False)

        def save_rule_wrapper(row):
            if isinstance(output_rule, str):
                group = output_rule
            elif callable(output_rule):
                group = output_rule(row)
                if group is None:
                    return None
                if not isinstance(group, str):
                    raise TypeError("String Expected.\nThe function passed to `output_rule` "
                                    "(`{0}`)\nreturned an object of type "
                                    "'{1}'.".format(output_rule.__name__, type(group).__name__))
            if action == 'copy':
                if os.path.isdir(group):
                    return group
                elif create_dirs and allow_write:
                    os.makedirs(group)
                    return group
                elif allow_write:
                    raise NotADirectoryError("\nNo such directory:\n'{0}'\n"
                                             "Consider setting `create_dirs=True`.".format(group))
            elif action == 'ndarray':
                return group

        return_dict = defaultdict(list)
        for _, row in tqdm(self.image_dataframe_cleaned.iterrows(), desc='Rendering Images',
                           disable=not status, total=len(self.image_dataframe_cleaned)):
            save_target = save_rule_wrapper(row)
            if isinstance(save_target, str):
                full_save_path = os_join(save_target, os_basename(row['cached_images_path']))
                if action == 'copy':
                    if allow_overwrite and allow_write:
                        row['cleaned_image'].save(full_save_path)
                    elif not os.path.isfile(full_save_path) and allow_write:
                        row['cleaned_image'].save(full_save_path)
                    return_dict[os_basename(save_target)] += [full_save_path]
                elif action == 'ndarray':
                    if ndarray_with_path:
                        return_dict[os_basename(save_target)] += [(np.array(row['cleaned_image']), full_save_path)]
                    else:
                        return_dict[os_basename(save_target)] += [np.array(row['cleaned_image'])]

        if action == 'ndarray':
            return {k: np.array(v) if not ndarray_with_path else v
                    for k, v in return_dict.items()}
        elif action == 'copy':
            return dict(return_dict)
