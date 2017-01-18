"""

    Image Processing
    ~~~~~~~~~~~~~~~~

"""
# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from PIL import ImageStat
from scipy.misc import imread

# General tools
from biovida.support_tools.support_tools import items_null

# Tools form the image subpackage
import biovida.images.resources # needed to generate medpix logo.
from biovida.images.openi_interface import OpenInterface
from biovida.images.image_tools import load_and_scale_imgs

from biovida.images.models.temp import cnn_model_location, resources_path # ToDo: remove.

# Models
from biovida.images.models.border_detection import border_detection
from biovida.images.models.img_classification import ImageRecognitionCNN
from biovida.images.models.template_matching import robust_match_template

# Tool to create required caches
from biovida.support_tools._cache_management import _package_cache_creator

pd.options.mode.chained_assignment = None  # Suppress Pandas' SettingWithCopyWarning

# ---------------------------------------------------------------------------------------------
# General procedure (for Ultrasound, X-ray, CT and MRI):
# ---------------------------------------------------------------------------------------------
#
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
#
#
# ---------------------------------------------------------------------------------------------
# grayscale Analysis
# ---------------------------------------------------------------------------------------------


class ImageProcessing(object):
    """

    This class is designed to allow easy analysis of cached image data.

    """

    def __init__(self, image_dataframe, model_location=None, verbose=True):
        """

        :param image_dataframe:
        :param model_location:
        """
        self.image_dataframe = image_dataframe
        self._verbose = verbose # ToDo: build out.

        # Start tqdm
        tqdm.pandas("status")

        # Load the CNN
        self.ircnn = ImageRecognitionCNN()

        # Load the model weights and architecture.
        if model_location is None:
            self.ircnn.load(cnn_model_location, override_existing=True)
        elif not isinstance(model_location, str):
            raise ValueError("`model_location` must either be a string or `None`.")
        elif os.path.isfile(model_location):
            self.ircnn.load(cnn_model_location, override_existing=True)
        else:
            raise FileNotFoundError("'{0}' could not be located.".format(str(model_location)))

        # Container for images represented as `ndarrays`
        self._ndarrays_images = None

    def _ndarray_extract(self, zip_with_column=None, reload_override=False):
        """

        :return:
        """
        if self._ndarrays_images is None or reload_override:
            self._ndarrays_images = [imread(i, flatten=True) for i in self.image_dataframe['img_cache_path']]

        if zip_with_column is not None:
            return zip(*[self._ndarrays_images, self.image_dataframe[zip_with_column]])
        else:
            return self._ndarrays_images

    def _grayscale_img(self, img_path):
        """

        Tool which uses the PIL library to determine whether or not an image is grayscale.
        Note:
              - this tool is very conservative (*any* 'color' will yield `False`).
              - the exception to the above rule is the *very rare* of an image which even split
                of red, green and blue.

        :param img_path: path to an image
        :type img_path:``str``
        :return: ``True`` if grayscale, else ``False``.
        :rtype: ``bool``
        """
        # See: http://stackoverflow.com/q/23660929/4898004
        if img_path is None or items_null(img_path):
            return np.NaN
        img = Image.open(img_path) # ToDo: Find a way to replace this.
        stat = ImageStat.Stat(img.convert("RGB"))
        return np.mean(stat.sum) == stat.sum[0]

    def grayscale_analysis(self, new_analysis=False):
        """

        Run a grayscale analysis.

        :param new_analysis:
        :return:
        """
        if 'grayscale' not in self.image_dataframe.columns or new_analysis:
            self.image_dataframe['grayscale'] = self.image_dataframe['img_cache_path'].progress_map(
                self._grayscale_img, na_action='ignore')

    def _logo_analysis_out(self, match, base_img_shape, output_params):
        """

        Decides the output for the `logo_analysis` function.

        :param match:
        :param output_params: tuple of the form:
                              (threshold, x_greater_check, y_greater_check, return_full).
        :return:
        """
        threshold, x_greater_check, y_greater_check, return_full = output_params

        # Check match quality
        if not isinstance(match, dict) or match['match_quality'] < threshold:
            return np.NaN

        # Check the box is in the top right
        if match['box']['bottom_left'][0] < (base_img_shape[0] * x_greater_check) or \
                match['box']['bottom_left'][1] > (base_img_shape[1] * y_greater_check):
            return np.NaN

        if return_full:
            return match['box']
        else:
            return match['box']['bottom_left']

    def logo_analysis(self,
                      threshold=0.25,
                      x_greater_check=1/3.0,
                      y_greater_check=1/2.5,
                      return_full=False,
                      new_analysis=False):  # ToDo: expose all `robust_match_template()` options.
        """

        Wraps ``biovida.images.models.template_matching.robust_match_template()``.

        :param threshold:
        :type threshold: ``float``
        :param x_greater_check:
        :type x_greater_check: ``float``
        :param y_greater_check:
        :type y_greater_check: ``float``
        :param return_full: if ``True``, return a dictionary with the location of all four corners for the logo's bounding box.
                            Otherwise, only the bottom left corner will be returned. Defaults to ``False``.
        :type return_full: ``bool``
        :return: the bottom left corner of
        :rtype: ``tuple``
        """
        if 'medpix_logo_lower_left' in self.image_dataframe.columns and not new_analysis:
            return None

        # Package Params
        output_params = (threshold, x_greater_check, y_greater_check, return_full)

        # Load the Pattern. ToDo: Generalize `resources_path` location.
        medline_template_img = imread(os.path.join(resources_path, "medpix_logo.png"), flatten=True)

        to_analyze = self._ndarray_extract(zip_with_column='journal_title')

        results = list()
        for img, journal in tqdm(to_analyze):
            if 'medpix' not in str(journal).lower():
                results.append(np.NaN)
            else:
                match, base_img_shape = robust_match_template(medline_template_img, img)
                current = self._logo_analysis_out(match, base_img_shape, output_params)
                results.append(current)

        # Update dataframe
        self.image_dataframe['medpix_logo_lower_left'] = results

    def border_analysis(self
                        , signal_strength_threshold=0.25
                        , min_border_separation=0.15
                        , lower_bar_search_space=0.9
                        , new_analysis=False):
        """

        Wrapper for ``biovida.images.models.border_detection.border_detection()``.

        :param signal_strength_threshold:
        :param min_border_separation:
        :param lower_bar_search_space:
        :param report_signal_strength:
        :return:
        """
        if all(x in self.image_dataframe.columns for x in ['hbar', 'hborder', 'vborder']) and not new_analysis:
            return None

        def ba_func(image):
            return border_detection(image,
                                    signal_strength_threshold,
                                    min_border_separation,
                                    lower_bar_search_space,
                                    report_signal_strength=False,
                                    rescale_input_ndarray=True)

        to_analyze = self._ndarray_extract()

        # Run the analysis.
        border_analysis = [ba_func(i) for i in tqdm(to_analyze)]

        # Convert to a dataframe and return
        ba_df = pd.DataFrame(border_analysis).fillna(np.NaN)

        # Update datefame
        self.image_dataframe['hbar'] = ba_df['hbar']
        self.image_dataframe['hborder'] = ba_df['hborder']
        self.image_dataframe['vborder'] = ba_df['vborder']

    def _h_crop_top_decision(self, x):
        """

        Choose lowest horizontal cropping point.
        Solves: upper 'hborder' vs 'medpix_logo_lower_left'.

        :return:
        """
        # Note: hborder = [top, lower]; medpix_logo_lower_left = [x, y].
        cols = ('hborder', 'medpix_logo_lower_left')
        crop_candiates = [x[i][0] if i == 'hborder' else x[i][1] for i in cols if not items_null(x[i])]
        return max(crop_candiates) if len(crop_candiates) else np.NaN

    def _h_crop_lower_decision(self, x):
        """

        Chose the highest cropping point for the bottom.
        Solves: lower 'hborder' vs 'hbar'.

        :param x:
        :return:
        """
        # Note: hborder = [top, lower]; hbar = int.
        lhborder = [x['hborder'][1]] if not items_null(x['hborder']) else []
        hbar = [x['hbar']] if not items_null(x['hbar']) else []
        crop_candiates = lhborder + hbar
        return min(crop_candiates) if len(crop_candiates) else np.NaN

    def crop_decision(self, new_analysis=False):
        """

        :return:
        """
        if all(x in self.image_dataframe.columns for x in ['upper_crop', 'lower_crop']) and not new_analysis:
            return None

        for i in ('medpix_logo_lower_left', 'hborder', 'hbar'):
            if i not in self.image_dataframe.columns.str.lower():
                raise AttributeError("The `image_dataframe` does not contain the\nfollowing required column: '{0}'.\n"
                                     "Please execute the corresponding analysis method to generate it.".format(i))

        # Compute Crop location
        self.image_dataframe['upper_crop'] = self.image_dataframe.apply(self._h_crop_top_decision, axis=1)
        self.image_dataframe['lower_crop'] = self.image_dataframe.apply(self._h_crop_lower_decision, axis=1)

    def _apply_cropping(self, img_cache_path, lower_crop, upper_crop, vborder, return_as_array=True):
        """

        :param img_cache_path:
        :param lower_crop:
        :param upper_crop:
        :param vborder:
        :param return_as_array:
        :return:
        :rtype: ``2D ndarray`` or ``PIL``
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

        if return_as_array:
            return np.asarray(converted_image.convert("RGB"))
        else:
            return converted_image.convert("RGB")

    def _image_problems_predictions(self, status=True):
        """

        This method is powered by a Convolutional Neural Network which
        computes probabilities for the presence of problematic image properties or types.

        Currently, the model can identify the follow problems:

            - arrows in images
            - images arrayed as grids

        :Examples:

        >>> df['img_problems']
        ...
        ...0   [(arrows, 0.197112), (grids, 0.0109332)]
        ...1   [(arrows, 0.211948), (grids, 0.00918275)]
        ...2   [(arrows, 0.473652), (grids, 0.00578115)]
        ...3   [(arrows, 0.43337),  (grids, 0.00857231)]
        ...4   [(grids, 0.928362), (arrows, 1.10526e-06)]

        The first value in the tuple represents the problem identified and second
        value represents its associated probability (for the sake of *simplicity*,
        this be interpreted as the model's confidence).

        For example, in the final row we can see that the model strongly 'believes' both
        that the image is, in fact, a grid of images and that it does not contain arrows.
        """
        # Zip the relevant columns (faster than looping through the dataframe directly).
        to_predict = zip(*[self.image_dataframe[i] for i in ('img_cache_path', 'lower_crop', 'upper_crop', 'vborder')])

        cropped_images_for_analysis = list()
        for img_cache_path, lower_crop, upper_crop, vborder in tqdm(to_predict):
            cropped_image = self._apply_cropping(img_cache_path, lower_crop, upper_crop, vborder, return_as_array=True)
            cropped_images_for_analysis.append(cropped_image)

        # Transform the cropped images into a form `ImageRecognitionCNN.predict()` can accept
        transformed_images = load_and_scale_imgs(cropped_images_for_analysis, self.ircnn.img_shape)

        # Make the predictions and Save
        self.image_dataframe['img_problems'] = self.ircnn.predict([transformed_images], status=status)

    def img_problems(self, status=True, new_analysis=False):
        """

        :param status:
        :param new_analysis:
        :return:
        """
        if 'img_problems' not in self.image_dataframe.columns or new_analysis:
            self._image_problems_predictions(status=status)

    def _auto_analysis(self, status=True, new_analysis=False):
        """

        :param status:
        :type status: ``bool``
        :param return_result:
        :type return_result: ``bool``
        :param new_analysis: recompute all analyses.

                        ..warning:

                            This will destroy the information currently computed and represented in the dataframe.

        :type new_analysis: ``bool``
        :return:
        :rtype: ``Pandas DataFrame
        """
        # Run Analysis Battery with Default Paramater Values
        self.grayscale_analysis(new_analysis)
        self.logo_analysis()
        self.border_analysis()

        # Compute Crop location
        self.crop_decision(new_analysis)

        # Generate predictions
        self.img_problems(status=True, new_analysis=False)

    def _auto_decision(self, threshold, require_grayscale):
        """

        :param threshold:
        :param require_grayscale:
        :return:
        """
        # ToDo: add variable threshold (e.g., arrows and grids).
        def reject_decision(x):
            if x['grayscale'] is not None and items_null(x['grayscale']) and \
                    x['grayscale'] == False and require_grayscale:
                    return True
            else:
                return any(x[1][1] > threshold for x in x['img_problems'])

        self.image_dataframe['reject'] = self.image_dataframe.apply(reject_decision, axis=1)

    def auto(self, threshold, require_grayscale, status=True, return_result=True, new_analysis=False):
        """

        :param threshold:
        :param require_grayscale:
        :param status:
        :param return_result:
        :param new_analysis:
        :return:
        """
        # Run Auto Analysis
        self._auto_analysis(status=True, new_analysis=False)

        # Run Auto Decision
        self._auto_decision(threshold, require_grayscale)

    def save(self, save_path):
        """

        Save processed images

        :return:
        """
        self.image_dataframe[self.image_dataframe['reject'] == True]

        to_predict = zip(*[self.image_dataframe[i] for i in ('img_cache_path', 'lower_crop', 'upper_crop', 'vborder')])
        for img_cache_path, lower_crop, upper_crop, vborder in tqdm(to_predict):
            # Apply Crop
            cropped_image = self._apply_cropping(img_cache_path, lower_crop, upper_crop, vborder, return_as_array=False)
            # Save to disk
            cropped_image.save(os.path.join(save_path, img_cache_path.split("/")[-1]))

































