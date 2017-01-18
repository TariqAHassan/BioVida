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

# Start tqdm
tqdm.pandas("status")


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
# Grayscale Analysis
# ---------------------------------------------------------------------------------------------


class ImageProcessing(object):
    """

    """
    def __init__(self, image_data_frame):
        self.image_data_frame = image_data_frame

        # Load the CNN
        self.ircnn = ImageRecognitionCNN()
        self.ircnn.load(cnn_model_location, override_existing=True)

    def _grayscale_img(self, img_path):
        """

        Tool which uses the PIL library to determine whether or not an image is grayscale.
        Note:
              - this tool is very conservative (*any* 'color' will yeild `False`).
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
        img = Image.open(img_path)
        stat = ImageStat.Stat(img.convert("RGB"))
        return np.mean(stat.sum) == stat.sum[0]

    # ToDo: replace 'x'.
    def require_grayscale_check(self, grayscale_tech=('mri', 'ct', 'x_ray', 'ultrasound')):
        """

        Check if image classification as an 'mri', 'pet', 'ct' is 'ultrasound'
        is valid based on whether or not the image is greyscale.

        :param x:
        :type x:
        :param grayscale_tech: a list of technologies that require greyscale images.
        :type grayscale_tech: ``tuple``
        :return:
        :rtype: ``bool``
        """
        # ToDo: this needs more sophisticated data that considers all available information.
        if x['image_modality_major'] in grayscale_tech and x['grayscale_img'] is False:
            return np.NaN
        else:
            return x['image_modality_major']

    def _logo_analysis_output(self, match, threshold, base_img_shape, x_greater_check, y_greater_check, return_full):
        """

        Decides the output for the `logo_analysis` function.

        :param match:
        :param threshold:
        :param base_img_shape:
        :param x_greater_check:
        :param y_greater_check:
        :param return_full:
        :return:
        """
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
                      x,
                      threshold=0.25,
                      x_greater_check=1/3.0,
                      y_greater_check=1/2.5,
                      return_full=False): # ToDo: expose all `robust_match_template()` options.
        """

        Wraps ``biovida.images.models.template_matching.robust_match_template()``.

        :param x: passed from df.apply().
        :type x: ``Pandas Series``
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
        if 'medpix' not in x['journal_title'].lower():
            return np.NaN

        # Load the Pattern
        medline_template_img = os.path.join(resources_path, "medpix_logo.png")

        # Run the matching algorithm
        match, base_img_shape = robust_match_template(medline_template_img, x['img_cache_path'])
        return self._logo_analysis_output(match, threshold, base_img_shape, x_greater_check, y_greater_check, return_full)

    def border_analysis(self
                         , join=True
                         , signal_strength_threshold=0.25
                         , min_border_separation=0.15
                         , lower_bar_search_space=0.9
                         , report_signal_strength=False):
        """

        Wrapper for ``biovida.images.models.border_detection.border_detection()``.

        :param data_frame:
        :param join: if ``True``, left join with ``data_frame``. Defaults to True.
        :param signal_strength_threshold:
        :param min_border_separation:
        :param lower_bar_search_space:
        :param report_signal_strength:
        :return:
        """
        def ba_func(image):
            return border_detection(image,
                                    signal_strength_threshold,
                                    min_border_separation,
                                    lower_bar_search_space,
                                    report_signal_strength)

        # Run the analysis
        border_analysis = data_frame['img_cache_path'].progress_map(ba_func)

        # Convert to a dataframe and return
        ba_df = pd.DataFrame(border_analysis.tolist()).fillna(np.NaN)
        return data_frame.join(ba_df, how='left') if join else ba_df

    def h_crop_top_decision(self, x):
        """

        Choose lowest horizontal cropping point.
        Solves: upper 'hborder' vs 'medpix_logo_lower_left'.

        :return:
        """
        # Note: hborder = [top, lower]; medpix_logo_lower_left = [x, y].
        cols = ('hborder', 'medpix_logo_lower_left')
        crop_candiates = [x[i][0] if i == 'hborder' else x[i][1] for i in cols if not items_null(x[i])]
        return max(crop_candiates) if len(crop_candiates) else np.NaN

    def h_crop_lower_decision(self, x):
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

    def apply_cropping(self, img_cache_path, lower_crop, upper_crop, vborder):
        """

        :param img_cache_path:
        :param lower_crop:
        :param upper_crop:
        :param vborder:
        :return:
        :rtype: ``2D ndarray``
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

        return np.asarray(converted_image.convert("RGB"))

    def image_problems_predictions(data_frame, ircnn, join=False, status=True):
        """

        Uses a Convolutional Neural Network to Make Predictions about the
        likelihood of problems in the image.

        Currently, the model can identify the follow problems

            - arrows in images
            - images arrayed as grids

        :param data_frame:
        :type data_frame:
        :param ircnn: an instance of biovida.images.models.img_classification.ImageRecognitionCNN().
        :type ircnn: ``ImageRecognitionCNN()``
        :param join:
        :type join:
        :return:
        :rtype: ``Pandas DataFrame`` or ``Pandas Series``

        :Examples:

        >>> df['img_problems']
        ...
        ...0   [(arrows, 0.197112), (grids, 0.0109332)]
        ...1   [(arrows, 0.211948), (grids, 0.00918275)]
        ...2   [(arrows, 0.473652), (grids, 0.00578115)]
        ...3   [(arrows, 0.43337),  (grids, 0.00857231)]
        ...4   [(grids, 0.928362), (arrows, 1.10526e-06)]

        The first value in the tuple represents the problem identified and second
        value represents its associated probability
        (for the sake of *simplicity*, this be interpreted as the model's confidence).

        For example, in the final row we can see that the model confidently 'believes' both
        that the image is, in fact, a grid of images and that it does not contain arrows.
        """
        # Zip the relevant columns (faster than looping through the dataframe directly).
        to_predict = zip(*[data_frame[i] for i in ('img_cache_path', 'lower_crop', 'upper_crop', 'vborder')])

        cropped_images_for_analysis = list()
        for img_cache_path, lower_crop, upper_crop, vborder in tqdm(to_predict):
            # Load the image and apply cropping
            cropped_image = apply_cropping(img_cache_path, lower_crop, upper_crop, vborder)
            cropped_images_for_analysis.append(cropped_image)

        # Transform the cropped images into a form `ImageRecognitionCNN.predict()` can accept
        transformed_images = load_and_scale_imgs(cropped_images_for_analysis, ircnn.img_shape)

        # Make the predictions
        predictions = ircnn.predict([transformed_images], status=status)

        if join:
            data_frame['img_problems'] = predictions
            return data_frame
        else:
            return pd.Series(predictions)

    def auto_image_clean(self, status=True):
        """

        :param status:
        :return:
        """
        # Compute whether or not the image is grayscale
        data_frame['grayscale'] = data_frame['img_cache_path'].progress_map(grayscale_img, na_action='ignore')

        # Report the location of the MedPix(R) logo (if present).
        data_frame['medpix_logo_lower_left'] = data_frame.progress_apply(self.logo_analysis, axis=1)

        # Analyze Crop Locations
        data_frame = border_analysis(data_frame)

        # Compute Crop location
        data_frame['upper_crop'] = data_frame.apply(self.h_crop_top_decision, axis=1)
        data_frame['lower_crop'] = data_frame.apply(self.h_crop_lower_decision, axis=1)

        # Generate predictions
        data_frame = image_problems_predictions(data_frame, ircnn, join=True, status=status)

        return data_frame
















































