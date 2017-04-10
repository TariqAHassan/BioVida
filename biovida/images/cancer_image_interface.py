# coding: utf-8

"""

    Cancer Imaging Archive Interface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import io
import os
import pickle
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd
from PIL import Image
from time import sleep
from copy import deepcopy
from warnings import warn
from datetime import datetime
from math import log10, floor

from biovida import __version_numeric__

# General Image Support Tools
from biovida.images._image_tools import NoResultsFound

# Database Management
from biovida.images.image_cache_mgmt import (_records_db_merge,
                                             _record_update_dbs_joiner,
                                             _prune_rows_with_deleted_images)

# General Support Tools
from biovida.support_tools.support_tools import (cln,
                                                 tqdm,
                                                 dicom,
                                                 header,
                                                 items_null,
                                                 only_numeric,
                                                 combine_dicts,
                                                 camel_to_snake_case,
                                                 data_frame_col_drop,
                                                 list_to_bulletpoints,
                                                 IN_NOTEBOOK)

# Import Printing Tools
from biovida.support_tools.printing import pandas_pprint

# Cache Management
from biovida.support_tools._cache_management import package_cache_creator

# Interface Support tools
from biovida.images._interface_support.shared import save_records_db
from biovida.images._interface_support.dicom_data_to_dict import dicom_to_dict
from biovida.images._interface_support.cancer_image.cancer_image_parameters import CancerImageArchiveParams
from biovida.images._interface_support.cancer_image.cancer_image_support_tools import nonessential_cancer_image_columns


# ----------------------------------------------------------------------------------------------------------
# Summarize Studies Provided Through the Cancer Imaging Archive
# ----------------------------------------------------------------------------------------------------------


class _CancerImageArchiveOverview(object):
    """

    Overview of Information Available on the Cancer Imaging Archive.

    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                        one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    :param verbose: if ``True`` print additional information. Defaults to ``False``.
    :type verbose: ``bool``
    :param tcia_homepage: URL to the the Cancer Imaging Archive's homepage.
    :type tcia_homepage: ``str``
    """
    # ToDo: add link to the TCIA page for a given collection/study (use BeautifulSoup).
    def __init__(self,
                 dicom_modality_abbrevs,
                 verbose=False,
                 cache_path=None,
                 tcia_homepage='http://www.cancerimagingarchive.net'):
        self._verbose = verbose
        self._tcia_homepage = tcia_homepage
        _, self._created_image_dirs = package_cache_creator(sub_dir='images', cache_path=cache_path,
                                                            to_create=['tcia'], nest=[('tcia', 'databases')],
                                                            verbose=verbose)

        self.dicom_modality_abbrevs = dicom_modality_abbrevs

    def _all_studies_parser(self):
        """

        Get a record of all studies on the Cancer Imaging Archive.

        :return: the table on the homepage
        :rtype: ``Pandas DataFrame``
        """
        # Extract the main summary table from the home page
        summary_df = pd.read_html(str(requests.get(self._tcia_homepage).text), header=0)[0]

        # Convert column names from camelCase to snake_cake
        summary_df.columns = list(map(camel_to_snake_case, summary_df.columns))

        summary_df = summary_df[summary_df['status'].str.strip().str.lower() != 'coming soon']
        summary_df = summary_df[~summary_df['location'].str.lower().str.contains('phantom')]
        summary_df = summary_df[~summary_df['collection'].str.lower().str.contains('mouse|phantom')]
        summary_df = summary_df[summary_df['access'].str.strip().str.lower() == 'public'].reset_index(drop=True)

        # Add Full Name for Modalities
        summary_df['modalities_full'] = summary_df['modalities'].map(
            lambda x: [self.dicom_modality_abbrevs.get(cln(i), i) for i in cln(x).split(", ")])

        # Parse the Location Column (and account for special case: 'Head-Neck').
        summary_df['location'] = summary_df['location'].map(
            lambda x: cln(x.replace(" and ", ", ").replace("Head-Neck", "Head, Neck")).split(", "))

        summary_df['updated'] = pd.to_datetime(summary_df['updated'], infer_datetime_format=True)
        summary_df.columns = list(map(lambda x: cln(x, extent=2), summary_df.columns))

        return summary_df

    def _obtain_tcia_overview(self, download_override=False):
        """

        Obtain and Manage a copy the table which summarizes the the Cancer Imaging Archive
        on the organization's homepage.

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
                                  Defaults to ``False``.
        :type download_override: ``bool``
        :return: summary table hosted on the home page of the Cancer Imaging Archive.
        :rtype: ``Pandas DataFrame``
        """
        # Define the path to save the data
        save_path = os.path.join(self._created_image_dirs['databases'], 'all_tcia_studies.p')

        if not os.path.isfile(save_path) or download_override:
            if self._verbose:
                header("Downloading Table of Available Studies... ", flank=False)
            summary_df = self._all_studies_parser()
            summary_df.to_pickle(save_path)
        else:
            summary_df = pd.read_pickle(save_path)

        return summary_df

    @staticmethod
    def _studies_filter(summary_df, cancer_type, location, modality):
        """

        Apply Filters passed to ``studies()``.

        :param summary_df: see: ``studies()``.
        :type summary_df: ``Pandas DataFrame``
        :param cancer_type: see: ``studies()``.
        :type cancer_type: ``str``, ``iterable`` or ``None``
        :param location: see: ``studies()``.
        :type location: ``str``, ``iterable`` or ``None``
        :param modality: see: ``studies()``.
        :type modality: ``str``, ``iterable`` or ``None``
        :return: ``summary_df`` with filters applied.
        :type: ``Pandas DataFrame``
        """
        # Filter by `cancer_type`
        if isinstance(cancer_type, (str, list, tuple)):
            if isinstance(cancer_type, (list, tuple)):
                cancer_type = "|".join(map(lambda x: cln(x).lower(), cancer_type))
            else:
                cancer_type = cln(cancer_type).lower()
            summary_df = summary_df[summary_df['cancer_type'].str.lower().str.contains(cancer_type)]

        # Filter by `location`
        if isinstance(location, (str, list, tuple)):
            location = [location] if isinstance(location, str) else location
            summary_df = summary_df[summary_df['location'].map(
                lambda x: any([cln(l).lower() in i.lower() for i in x for l in location]))]

        def modality_filter(x, modality):
            """Apply filter to look for rows which match `modality`."""
            sdf_modalities = cln(x['modalities']).lower()
            sdf_modalities_full = [cln(i).lower() for i in x['modalities_full']]

            if any(m in sdf_modalities for m in modality):
                return True
            if any([m in m_full for m in modality for m_full in sdf_modalities_full]):
                return True
            else:
                return False

        # Filter by `modality`.
        if isinstance(modality, (str, list, tuple)):
            modality = [modality.lower()] if isinstance(modality, str) else list(map(lambda x: x.lower(), modality))
            summary_df = summary_df[summary_df.apply(lambda x: modality_filter(x, modality), axis=1)]

        return summary_df


# ----------------------------------------------------------------------------------------------------------
# Pull Records from the Cancer Imaging Archive
# ----------------------------------------------------------------------------------------------------------


class _CancerImageArchiveRecords(object):
    """

    Class to harvest records for a given collection/study through the Cancer Imaging Archive API.

    :param api_key: an key to the the Cancer Imaging Archive's API.
    :type api_key: ``str``
    :param dicom_modality_abbrevs: an instance of ``CancerImageArchiveParams().dicom_modality_abbreviations('dict')``
    :type dicom_modality_abbrevs: ``dict``
    :param tcia_overview_df: yield of ``_CancerImageArchiveOverview()._obtain_tcia_overview()``
    :type tcia_overview_df: ``Pandas DataFrames``
    :param root_url: the root URL for the Cancer Imaging Archive's API.
    :type root_url: ``str``
    """

    def __init__(self, api_key, dicom_modality_abbrevs, tcia_overview_df, root_url):
        self.ROOT_URL = root_url
        self.records_df = None
        self.dicom_modality_abbrevs = dicom_modality_abbrevs
        self._tcia_overview_df = tcia_overview_df
        self.API_KEY = api_key
        self._url_sep = '+'

    def _study_extract(self, study):
        """

        Download all patients in a given study.

        :param study: a Cancer Imaging Archive collection (study)
        :type study: ``str``
        :return: the yield of passing the ``getPatientStudy`` parameter to the Cancer Imaging Archive API for a given
                 collection (study).
        :rtype: ``Pandas DataFrame``
        """
        url = '{0}/query/getPatientStudy?Collection={1}&format=csv&api_key={2}'.format(
            self.ROOT_URL, cln(study).replace(' ', self._url_sep), self.API_KEY)
        data_frame = pd.DataFrame.from_csv(url).reset_index()

        # Convert column names from camelCase to snake_cake
        data_frame.columns = list(map(camel_to_snake_case, data_frame.columns))

        return data_frame

    def _robust_study_extract(self, study):
        """

        This method uses '+' first as a replacement for spaces when sending requests to the Cancer Imaging Archive.
        If that fails, this method falls back on '-'.

        :param study: a Cancer Imaging Archive collection (study).
        :type study: ``str``
        :return: see: ``_study_extract()``
        :rtype: ``Pandas DataFrame``

        :raises IndexError: if both '+' and '-' fail to yield a dataframe with nonzero length.
        """
        study_df = self._study_extract(study)
        if study_df.shape[0] == 0:
            self._url_sep = '-'
            study_df = self._study_extract(study)
            self._url_sep = '+'  # reset
            if study_df.shape[0] == 0:
                raise IndexError("The '{0}' collection/study data has no length.\n"
                                 "This is likely the result of a problem with the "
                                 "Cancer Imaging Archive REST API's.\n".format(study))
        return study_df

    @staticmethod
    def _date_index_map(list_of_dates):
        """

        Returns a dict of the form: ``{date: index in ``list_of_dates``, ...}``

        :param list_of_dates: a list (or tuple) of datetime objects.
        :type list_of_dates: ``list`` or ``tuple``
        :return: description (above)
        :rtype: ``dict``
        """
        return {k: i for i, k in enumerate(sorted(list_of_dates), start=1)}

    def _summarize_study_by_patient(self, study):
        """

        Summarizes a study by patient.

        :param study: a Cancer Imaging Archive collection (study).
        :type study: ``str``
        :return: nested dictionary of the form:

                ``{patient_id: {study_instance_uid: {'sex': ..., 'age': ..., 'session': ..., 'study_date': ...}}}``

        :rtype: ``dict``
        """
        # Download a summary of all patients in a study
        study_df = self._robust_study_extract(study)

        study_df['study_date'] = pd.to_datetime(study_df['study_date'], infer_datetime_format=True)

        # Divide Study into stages (e.g., Baseline (session 1); Baseline + 1 Month (session 2), etc.
        stages = study_df.groupby('patient_id').apply(
            lambda x: self._date_index_map(x['study_date'].tolist())).to_dict()

        # Apply stages
        study_df['session'] = study_df.apply(lambda x: stages[x['patient_id']][x['study_date']], axis=1)

        # Define columns to extract from `study_df`
        valuable_cols = ('patient_id', 'study_instance_uid', 'session', 'patient_sex', 'patient_age', 'study_date')

        # Convert to a nested dictionary
        patient_dict = dict()
        for pid, si_uid, session, sex, age, date in zip(*[study_df[c] for c in valuable_cols]):
            inner_nest = {'sex': sex, 'age': age, 'session': session, 'study_date': date}
            if pid not in patient_dict:
                patient_dict[pid] = {si_uid: inner_nest}
            else:
                patient_dict[pid] = combine_dicts(patient_dict[pid], {si_uid: inner_nest})

        return patient_dict

    def _patient_image_summary(self, patient, study, patient_dict):
        """

        Harvests the Cancer Image Archive's Text Record of all baseline images for a given patient
        in a given study.

        :param patient: the patient_id (will be used to form the request to the TCIA server).
        :type patient: ``str``
        :param study: a Cancer Imaging Archive collection (study).
        :type study: ``str``
        :param patient_dict: a value in ``study_dict`` (which is a dictionary itself).
        :type patient_dict: ``dict``
        :return: the yield of the TCIA ``getSeries`` param for a given patient in a given collection (study).
                 Their sex, age, the session number (e.g., baseline = 1, baseline + 1 month = 2, etc.) and the
                 'study_date' (i.e., the date the study was conducted).
        :rtype: ``Pandas DataFrame``
        """
        # Select an individual Patient
        url = '{0}/query/getSeries?Collection={1}&PatientID={2}&format=csv&api_key={3}'.format(
            self.ROOT_URL, cln(study).replace(' ', self._url_sep), patient, self.API_KEY)
        patient_df = pd.DataFrame.from_csv(url).reset_index()

        # Convert column names from camelCase to snake_cake
        patient_df.columns = list(map(camel_to_snake_case, patient_df.columns))

        # Add sex, age, session, study_date and patient id
        patient_info = patient_df['study_instance_uid'].map(
            lambda x: {k: patient_dict[x][k] for k in ('sex', 'age', 'session', 'study_date')})
        patient_df = patient_df.join(pd.DataFrame(patient_info.tolist()))
        patient_df['patient_id'] = patient

        return patient_df

    def _clean_patient_study_df(self, patient_study_df):
        """

        Cleans the input in the following ways:

            - convert 'F' --> 'Female' and 'M' --> 'Male'

            - Converts the 'age' column to numeric (years)

            - Remove line breaks in the 'protocol_name' and 'series_description' columns

            - Add Full name for modality (modality_full)

            - Convert the 'series_date' column to datetime

        :param patient_study_df: the ``patient_study_df`` dataframe evolved inside ``_pull_records()``.
        :type patient_study_df: ``Pandas DataFrame``
        :return: a cleaned ``patient_study_df``
        :rtype: ``Pandas DataFrame``
        """
        # convert 'F' --> 'female' and 'M' --> 'male'.
        patient_study_df['sex'] = patient_study_df['sex'].map(
            lambda x: {'F': 'female', 'M': 'male'}.get(cln(str(x)).upper(), x),
            na_action='ignore')

        # Convert entries in the 'age' Column to floats.
        patient_study_df['age'] = patient_study_df['age'].map(
            lambda x: only_numeric(x) / 12.0 if 'M' in str(x).upper() else only_numeric(x),
            na_action='ignore')

        # Remove unneeded line break marker
        for c in ('protocol_name', 'series_description', 'manufacturer_model_name'):
            patient_study_df[c] = patient_study_df[c].map(
                lambda x: cln(x.replace("\/", " ")) if isinstance(x, str) else x, na_action='ignore')

        # Add the full name for modality.
        patient_study_df['modality_full'] = patient_study_df['modality'].map(
            lambda x: self.dicom_modality_abbrevs.get(x, np.NaN), na_action='ignore')

        # Lower and clean 'body_part_examined' column
        patient_study_df['body_part_examined'] = patient_study_df['body_part_examined'].map(
            # ToDo: generalize the special case used here to handle 'headneck'.
            lambda x: cln(x).lower().replace("headneck", "head, neck") if isinstance(x, str) else x,
            na_action='ignore')

        patient_study_df['series_date'] = pd.to_datetime(patient_study_df['series_date'],
                                                         infer_datetime_format=True)

        def series_number_rescale(x):
            if isinstance(x, (int, float)) and not items_null(x) and x > 0:
                return x / 10**floor(log10(x))
            else:
                return x

        patient_study_df['series_number_rescaled'] = patient_study_df['series_number'].map(
            series_number_rescale, na_action='ignore')

        sort_by = ['patient_id', 'session', 'series_number_rescaled']
        return patient_study_df.fillna(np.NaN).sort_values(by=sort_by).reset_index(drop=True)

    def _get_condition_name(self, collection_series):
        """

        This method gets the name of the condition studied for a given collection (study).
        (collection_series gets reduced down to a single unique).

        :param collection_series: a series of the study name, e.g., ('MY-STUDY', 'MY-STUDY', 'MY-STUDY', ...).
        :type collection_series: ``Pandas Series``
        :return: the name of disease studied in a given collection (lower case).
        :rtype: ``str``
        """
        unique_studies = collection_series.unique()
        if len(unique_studies) == 1:
            collection = unique_studies[0]
        else:
            raise AttributeError("`{0}` studies found in `records`. "
                                 "Expected one.".format(str(len(unique_studies))))

        condition_name = self._tcia_overview_df[
            self._tcia_overview_df['collection'] == collection]['cancer_type'].iloc[0]

        return condition_name.lower() if isinstance(condition_name, str) else condition_name

    def records_pull(self, study, search_dict, pull_time, patient_limit, verbose):
        """

        Extract record of all images for all patients in a given study.

        :param study: a Cancer Imaging Archive collection (study).
        :type study: ``str``
        :param search_dict: a dictionary which contains the search information provided by the user
                            (as evolved inside  ``CancerImageInterface()_search_dict_gen()``.
        :type search_dict: ``dict``
        :param pull_time: the time the query was launched.
        :type pull_time: ``datetime``
        :param patient_limit: limit on the number of patients to extract.
                              Patient IDs are sorted prior to this limit being imposed.
                              If ``None``, no `patient_limit` will be imposed.
        :type patient_limit: ``int`` or ``None``
        :param verbose: if ``True`` print additional information. Defaults to ``False``.
        :type verbose: ``bool``
        :return: a dataframe of all baseline images
        :rtype: ``Pandas DataFrame``
        """
        study_dict = self._summarize_study_by_patient(study)

        # Check for invalid `patient_limit` values:
        if not isinstance(patient_limit, int) and patient_limit is not None:
            raise ValueError('`patient_limit` must be an integer or `None`.')
        elif isinstance(patient_limit, int) and patient_limit < 1:
            raise ValueError('If `patient_limit` is an integer it must be greater than or equal to 1.')

        # Define number of patients to extract
        s_patients = sorted(study_dict.keys())
        patients_to_obtain = s_patients[:patient_limit] if isinstance(patient_limit, int) else s_patients

        # Evolve a dataframe ('frame') for the baseline images of all patients
        frames = list()
        for patient in tqdm(patients_to_obtain, desc="'{0}' Records".format(study), disable=not verbose):
            frames.append(self._patient_image_summary(patient, study=study, patient_dict=study_dict[patient]))

        # Concatenate baselines frame for each patient
        patient_study_df = pd.concat(frames, ignore_index=True)

        patient_study_df['article_type'] = ['case_report'] * patient_study_df.shape[0]
        patient_study_df['study_name'] = study
        patient_study_df['cancer_type'] = self._get_condition_name(patient_study_df['collection'])

        # Add the Search query which created the current results and the time the search was launched.
        patient_study_df['query'] = [search_dict] * patient_study_df.shape[0]
        patient_study_df['pull_time'] = [pull_time] * patient_study_df.shape[0]

        self.records_df = self._clean_patient_study_df(patient_study_df)

        return self.records_df


# ----------------------------------------------------------------------------------------------------------
# Pull Images from the Cancer Imaging Archive
# ----------------------------------------------------------------------------------------------------------


class _CancerImageArchiveImages(object):
    """

    Class to harvest images for a given collection/study through the Cancer Imaging Archive API, based on
    records extracted by ``_CancerImageArchiveRecords()``.

    :param api_key: an key to the the Cancer Imaging Archive's API.
    :type api_key: ``str``
    :param dicom_modality_abbrevs: an instance of ``CancerImageArchiveParams().dicom_modality_abbreviations('dict')``
    :type dicom_modality_abbrevs: ``dict``
    :param root_url: the root URL for the the Cancer Imaging Archive's API.
    :type root_url: ``str``
    :param verbose: if ``True`` print additional details.
    :type verbose: ``bool``
    :type cache_path: ``str`` or ``None``
    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    """

    def __init__(self, api_key, dicom_modality_abbrevs, root_url, verbose, cache_path=None):
        _, self._created_image_dirs = package_cache_creator(sub_dir='images',
                                                            cache_path=cache_path,
                                                            to_create=['tcia'],
                                                            nest=[('tcia', 'raw'), ('tcia', 'dicoms'),
                                                                  ('tcia', 'databases')],
                                                            verbose=verbose)

        self.verbose = verbose
        self.API_KEY = api_key
        self.dicom_modality_abbrevs = dicom_modality_abbrevs
        self.ROOT_URL = root_url

        # Add Record DataFrame; this database updates in real-time as the images are downloaded.
        self.records_db_images = None
        self.real_time_update_db = None

        # Define the path to the temporary directory
        self.temp_directory_path = os.path.join(self._created_image_dirs['databases'], "__temp__")

    def _instantiate_real_time_update_db(self, db_index):
        """

        Create the ``real_time_update_db`` and define the path to the location where it will be saved.

        :param db_index: the index of the ``real_time_update_db`` dataframe (should be from ``records_db``).
        :type db_index: ``Pandas Series``
        """
        real_time_update_columns = ['cached_dicom_images_path', 'cached_images_path', 'error_free_conversion',
                                    'allowed_modality', 'image_count_converted_cache']

        self.real_time_update_db = pd.DataFrame(
            columns=real_time_update_columns, index=db_index).replace({np.NaN: None})

    def _download_zip(self, series_uid, temporary_folder):
        """

        Downloads the zipped from from the Cancer Imaging Archive for a given 'SeriesInstanceUID' (``series_uid``).

        :param series_uid: the 'series_instance_uid' needed to use TCIA's ``getImage`` parameter
        :type series_uid: ``str``
        :param temporary_folder: path to the temporary folder where the images will be (temporary) cached.
        :type temporary_folder: ``str``
        :return: list of paths to the new files.
        :rtype: ``list``
        """
        # See: http://stackoverflow.com/a/14260592/4898004
        url = '{0}/query/getImage?SeriesInstanceUID={1}&format=csv&api_key={2}'.format(
            self.ROOT_URL, series_uid, self.API_KEY)
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(temporary_folder)

        def file_path_full(f):
            """Construct the full path for a given file in ``z``."""
            base_name = cln(os.path.basename(f.filename))
            return os.path.join(temporary_folder, base_name) if len(base_name) else None

        # Generate the list of paths to the dicoms
        return list(filter(None, map(file_path_full, z.filelist)))

    @staticmethod
    def _extract_instance_number(f):
        """
        
        Extract the instance number from a DICOM file.
        
        :param f: a DICOM file
        :param f: ``pydicom object``
        :return: instance number
        :rtype: ``str``
        """
        try:
            instance_number = f.InstanceNumber
        except:
            return 'NA'

        cleaned_instance_number = cln(str(instance_number))
        if len(cleaned_instance_number) and cleaned_instance_number.isdigit():
            return cleaned_instance_number
        else:
            return 'NA'

    def _dicom_to_standard_image(self, f, pull_position, series_uid, conversion, new_file_name):
        """

        This method handles the act of saving dicom images as in a more common file format (e.g., .png).
        An image (``f``) can be either 2 or 3 Dimensional.

        Notes:

        - 3D images will be saved as individual frames

        - if pydicom cannot render the DICOM as a pixel array, this method will its halt image extraction efforts.

        :param f: a (py)dicom image.
        :type f: ``pydicom object``
        :param pull_position: the position of the file in the list of files pulled from the database.
        :type pull_position: ``int``
        :param series_uid: the 'series_instance_uid' needed to use TCIA's ``getImage`` parameter
        :type series_uid: ``str``
        :param conversion: the color scale conversion to use, e.g., 'LA' or 'RGB'.
        :type conversion: ``str``
        :param new_file_name: see ``_convert_and_save_dicom``'s ``save_name`` parameter.
        :type new_file_name: ``str``
        :return: tuple of the form: ``(a list of paths to saved images, boolean denoting success)``
        :rtype: ``tuple``
        """
        # Note: f.PatientsWeight will extract weight from the dicom.

        # Define a list to populate with a record of all images saved
        all_save_paths = list()

        # Extract a pixel array from the dicom file.
        try:
            pixel_arr = f.pixel_array
        except (UnboundLocalError, TypeError):
            return [], False

        save_location = self._created_image_dirs['raw']

        def save_path(instance):
            """Define the path to save the image to."""
            head = "{0}_{1}".format(instance, pull_position)
            file_name = "{0}__{1}__default.{2}".format(
                head, os.path.basename(new_file_name), "png")
            return os.path.join(save_location, file_name)

        if pixel_arr.ndim == 2:
            # Define save name by combining the images instance in the set.
            path = save_path(self._extract_instance_number(f))
            Image.fromarray(pixel_arr).convert(conversion).save(path)
            all_save_paths.append(path)
        # If ``f`` is a 3D image (e.g., segmentation dicom files), save each layer as a separate file/image.
        elif pixel_arr.ndim == 3:
            for instance, layer in enumerate(range(pixel_arr.shape[0]), start=1):
                path = save_path(instance)
                Image.fromarray(pixel_arr[layer:layer + 1][0]).convert(conversion).save(path)
                all_save_paths.append(path)
        else:
            warn("\nProblem converting an image for\nthe following series_instance_uid:\n\n{0}\n\nCannot "
                 "stabilize {1} dimensional arrays.\nImages must be 2D or 3D.".format(series_uid, pixel_arr.ndim))
            return [], False

        return all_save_paths, True

    def _update_and_set_list(self, index, column, new, return_replacement_len=False):
        """

        Set = add to ``self.real_time_update_db``.
        Update = update the list (or tuple) already there.

        Note: the 'list' added to ``self.real_time_update_db`` will actually be a tuple.

        :param column: the name of the column in ``self.real_time_update_db`` to be updated.
        :type column: ``str``
        :param new: the new list to be added.
        :type new: ``list`` or ``tuple``
        :return: the length of the replacement, if ``return_replacement_len`` is ``True``
        :rtype: ``int`` or ``None``
        """
        current = self.real_time_update_db.get_value(index, column)

        def cleaner(to_clean):
            """Tool which will ensure the output is a `list`."""
            return list(to_clean) if isinstance(to_clean, (list, tuple)) else []

        # Clean `current` and `new`, combine and generate a replacement candidate
        replacement_candidate = tuple(cleaner(current) + cleaner(new))

        # Generate the replacement
        replacement = replacement_candidate if len(replacement_candidate) else np.NaN

        self.real_time_update_db.set_value(index, column, replacement)

        return len(replacement) if return_replacement_len and isinstance(replacement, (list, tuple)) else None

    def _convert_and_save_dicom(self,
                                path_to_dicom_file,
                                index,
                                pull_position,
                                series_uid,
                                save_name,
                                color=False):
        """

        Save a dicom image as a more common file format.

        :param path_to_dicom_file: path to a dicom image
        :type path_to_dicom_file: ``str``
        :param pull_position: the position of the image in the raw zip data provided by the Cancer Imaging Archive API.
        :type pull_position: ``int``
        :param series_uid: the 'series_instance_uid' needed to use TCIA's ``getImage`` parameter
        :type series_uid: ``str``
        :param index: the row index currently being processed inside of the main loop in ``_pull_images_engine()``.
        :type index: ``int``
        :param save_name: name of the new file (do *NOT* include a file extension).
                          If ``None``, name from ``path_to_dicom_file`` will be conserved.
        :type save_name: ``str``
        :param color: If ``True``, convert the image to RGB before saving. If ``False``, save as a grayscale image.
                      Defaults to ``False``
        :type color: ``bool``
        """
        # Load the DICOM file into RAM
        f = dicom.read_file(path_to_dicom_file)

        # Conversion (needed so the resultant image is not pure black)
        conversion = 'RGB' if color else 'LA'  # note: 'LA' = grayscale.

        if isinstance(save_name, str):
            new_file_name = save_name
        else:
            # Remove the file extension and then extract the base name from the path.
            new_file_name = os.path.basename(os.path.splitext(path_to_dicom_file)[0])

        # Convert the image into a PIL object and save to disk.
        all_save_paths, success = self._dicom_to_standard_image(f=f, pull_position=pull_position,
                                                                series_uid=series_uid, conversion=conversion,
                                                                new_file_name=new_file_name)

        # Update Record
        cfp_len = self._update_and_set_list(index, 'cached_images_path', all_save_paths, return_replacement_len=True)
        self.real_time_update_db.set_value(index, 'image_count_converted_cache', cfp_len)

        # Add record of whether or not the dicom file could be converted to a standard image type
        if self.real_time_update_db.get_value(index, 'error_free_conversion') != False:  # Block False being replaced.
            self.real_time_update_db.set_value(index, 'error_free_conversion', success)

    def _move_dicoms(self, dicom_files, series_abbrev, index):
        """

        Move the dicom source files to ``self._created_image_dirs['dicoms']``.
        Employ to prevent the raw dicom files from being destroyed.

        :param dicom_files: the yield of ``_download_zip()``
        :type dicom_files: ``list``
        :param series_abbrev: as evolved inside ``_pull_images_engine()``.
        :type series_abbrev: ``str``
        :param index: the row index currently being processed inside of the main loop in ``_pull_images_engine()``.
        :type index: ``int``
        """
        new_dicom_paths = list()
        for file in dicom_files:
            # Define a name for the new file by extracting the dicom file name and combining with `series_abbrev`.
            instance_number = self._extract_instance_number(f=dicom.read_file(file))
            file_parsed = list(os.path.splitext(os.path.basename(file)))
            new_dicom_file_name = "{0}_{1}__{2}{3}".format(
                # ToDo: this is redundant. The file is read twice: once to move and once if converting to PNG.
                instance_number, file_parsed[0], series_abbrev, file_parsed[1])

            new_location = os.path.join(self._created_image_dirs['dicoms'], new_dicom_file_name)
            new_dicom_paths.append(new_location)

            if os.path.isfile(new_location):
                os.remove(new_location)
            shutil.move(file, new_location)

        def sort_func(i):
            """Sort `new_dicom_paths` by instance_number."""
            try:
                instance_number = os.path.basename(i).split("_")[0]
                return float(instance_number)
            except:
                return len(new_dicom_paths)  # `len` is O(1)

        sorted_paths = tuple(sorted(new_dicom_paths, key=sort_func))
        self.real_time_update_db.set_value(index, 'cached_dicom_images_path', sorted_paths)

    def _cache_check(self, series_abbrev, n_images_min, save_png, save_dicom):
        """

        Check that caches likely contain that data which would be obtained by downloading it from the database.

        :param series_abbrev: as evolved inside ``_pull_images_engine()``
        :type series_abbrev: ``str``
        :param n_images_min: `image_count` as passed in ``_pull_images_engine()``. Denotes the min. number of images
                              for a given series for the cache to be considered complete (less than this number
                              will trigger an effort to download the corresponding images (or, more specifically,
                              SeriesInstanceUID). Note: This number gives the number of dicoms -- the number of
                              converted images could be larger (i.e., 3D images which can unpack to be many more),
                              but it is not possible to known this without actually downloading and unpacking these
                              images.
        :param save_png: ``True`` signals the user's intention to convert DICOMs to PNG.
        :type save_png: ``bool``
        :param save_dicom: see: ``pull_images()``.
        :type save_dicom: ``bool``
        :type n_images_min: ``int``
        :return: tuple of the form:

        ``(cache likely complete,
           series_abbrev matches in self._created_image_dirs['raw'],
           series_abbrev matches in self._created_image_dirs['dicoms'])``

        :rtype: ``tuple``
        """
        # Check that `self._created_image_dirs['raw']` has files which contain the string `series_abbrev`.
        converted_loc_summary = tuple(sorted([os.path.join(self._created_image_dirs['raw'], f)
                                              for f in os.listdir(self._created_image_dirs['raw'])
                                              if series_abbrev in f]))

        # Check that `self._created_image_dirs['dicoms'])` has files which contain the string `series_abbrev`.
        dicoms_sl_summary = tuple(sorted([os.path.join(self._created_image_dirs['dicoms'], f)
                                          for f in os.listdir(self._created_image_dirs['dicoms'])
                                          if series_abbrev in f]))

        if save_png:
            converted_loc_summary_complete = len(converted_loc_summary) >= n_images_min
        else:
            converted_loc_summary_complete = True  # i.e., the actual answer is irrelevant

        if save_dicom:
            dicoms_sl_summary_complete = len(dicoms_sl_summary) >= n_images_min
        else:
            dicoms_sl_summary_complete = True  # i.e., the actual answer is irrelevant

        cache_complete = converted_loc_summary_complete and dicoms_sl_summary_complete

        return (cache_complete,
                converted_loc_summary if len(converted_loc_summary) else None,
                dicoms_sl_summary if len(dicoms_sl_summary) else None)

    def _create_temp_dicom_dir(self):
        """

        Create a temporary directory for DICOM files immediately following download from
        The Cancer Imaging Archive.

        :return: the full path to the newly created temporary directory.
        :rtype: ``str``
        """
        temp_folder = os.path.join(self._created_image_dirs['dicoms'], '__temp_dicom__')

        if os.path.isdir(temp_folder):
            # To prevent duplicate images being created,
            # destroy this directory if it already exists.
            shutil.rmtree(temp_folder, ignore_errors=True)
        os.makedirs(temp_folder)
        return temp_folder

    @staticmethod
    def _valid_modality(allowed_modalities, modality, modality_full):
        """

        Check if `modality` or `modality_full` contains the modality the user is looking for.

        :param allowed_modalities: see: ``pull_images()``
        :type allowed_modalities: ``list``, ``tuple`` or ``None``.
        :param modality: a single element from the ``modality`` column in ``self.real_time_update_db``.
        :type modality: ``str``
        :param modality_full: a single element from the ``modality_full`` column in ``self.real_time_update_db``.
        :type modality_full: ``str``
        :return: whether or not the image satisfies the modality the user is looking for.
        :rtype: ``bool``
        """
        # Assume True if `allowed_modalities` is left to its default (`None`).
        if allowed_modalities is None:
            return True

        if not isinstance(allowed_modalities, (list, tuple, str)):
            raise ValueError("`allowed_modalities` must be one of `list`, `tuple`, `str`.")

        # Convert `allowed_modalities` to an iterable
        if isinstance(allowed_modalities, str):
            allowed_modalities = [allowed_modalities]

        # Check if any item in `allowed_modalities` is a sublist in `modality` or `modality_full`.
        if any([cln(l).lower() in cln(i).lower() for l in allowed_modalities for i in (modality, modality_full)]):
            return True
        else:
            return False

    def _pull_images_engine(self, save_dicom, allowed_modalities, save_png):
        """

        Tool to coordinate the above machinery for pulling and downloading images (or locating them in the cache).

        :param save_dicom: see: ``pull_images()``.
        :type save_dicom: ``bool``
        :param save_png: see: ``pull_images()``
        :param save_png: ``bool``
        :param allowed_modalities: see: ``pull_images()``
        :type allowed_modalities: ``list``, ``tuple`` or ``None``
        """
        for index, row in tqdm(self.records_db_images.iterrows(), total=len(self.records_db_images),
                               desc='Obtaining Images', disable=not self.verbose):
            # Check if the image should be harvested (or loaded from the cache).
            valid_image_modality = self._valid_modality(allowed_modalities, row['modality'], row['modality_full'])

            # Add whether or not the image was of the modality (or modalities) requested by the user.
            self.real_time_update_db.set_value(index, 'allowed_modality', valid_image_modality)

            # Compose central part of the file name from 'patient_id' and the last ten digits of 'series_instance_uid'
            series_abbrev = "{0}_{1}".format(row['patient_id'], str(row['series_instance_uid'])[-10:])

            # Analyze the cache to determine whether or not downloading the images is needed
            cache_complete, sl_summary, dsl_summary = self._cache_check(series_abbrev=series_abbrev,
                                                                        n_images_min=row['image_count'],
                                                                        save_png=save_png,
                                                                        save_dicom=save_dicom)

            if valid_image_modality and not cache_complete:
                temporary_folder = self._create_temp_dicom_dir()

                # Download the images into a temporary folder.
                dicom_files = self._download_zip(row['series_instance_uid'], temporary_folder=temporary_folder)

                if save_png:
                    for e, f in enumerate(dicom_files, start=1):
                        self._convert_and_save_dicom(path_to_dicom_file=f, index=index, pull_position=e,
                                                     series_uid=row['series_instance_uid'],
                                                     save_name=series_abbrev)

                if save_dicom:
                    self._move_dicoms(dicom_files=dicom_files, series_abbrev=series_abbrev, index=index)

                shutil.rmtree(temporary_folder, ignore_errors=True)
            else:
                self._update_and_set_list(index, 'cached_dicom_images_path', dsl_summary)
                self._update_and_set_list(index, 'cached_images_path', sl_summary)
                if save_png:  # otherwise, leave as None/NaN
                    self.real_time_update_db.set_value(index, 'error_free_conversion', cache_complete)
                converted_image_count = len(sl_summary) if isinstance(sl_summary, (list, tuple)) else None
                self.real_time_update_db.set_value(index, 'image_count_converted_cache', converted_image_count)

    def pull_images(self, records_db, session_limit, save_png, save_dicom, allowed_modalities):
        """

        Pull Images from the Cancer Imaging Archive.

        :param records_db: the yield from ``_CancerImageArchiveRecords().records_pull()``.
        :type records_db: ``Pandas DataFrame``
        :param session_limit: restrict image harvesting to the first ``n`` sessions, where ``n`` is the value passed
                              to this parameter. If ``None``, no limit will be imposed.
        :type session_limit: ``int``
        :param save_png: see ``CancerImageArchive().pull()``.
        :type save_png: ``bool``
        :param save_dicom: if ``True``, save the raw dicom files.
        :type save_dicom: ``bool``
        :param allowed_modalities: limit images downloaded to certain modalities.
                                   See: CancerImageInterface().dicom_modality_abbrevs (use the keys).
                                   Note: 'MRI', 'PET', 'CT' and 'X-Ray' can also be used.
                                   This parameter is not case sensitive.
        :type allowed_modalities: ``list``, ``tuple`` or ``None``
        :return: a dataframe with information about the images cached by this method.
        :rtype: ``Pandas DataFrame``
        """
        # Notes on 'image_count_converted_cache':
        # 1. a column which provides the number of images each SeriesInstanceUID yielded
        # 2. values may be discrepant with the 'image_count' column because 3D images are expanded
        #    into their individual frames when saved to the converted images cache.

        # Create the __temp__ folder for `image_pull_settings.p` if it does not already exist.
        if not os.path.isdir(self.temp_directory_path):
            os.makedirs(self.temp_directory_path)

        settings_path = os.path.join(self.temp_directory_path, "image_pull_settings.p")
        pickle.dump({k: v for k, v in locals().items() if k not in ('self', 'settings_path')},
                    open(settings_path, "wb"))

        if isinstance(session_limit, int):
            if session_limit < 1:
                raise ValueError("`session_limit` must be greater than or equal to 1.")
            self.records_db_images = records_db[records_db['session'].map(
                lambda x: float(x) <= session_limit if pd.notnull(x) else False)].reset_index(drop=True).copy(deep=True)
        else:
            self.records_db_images = records_db.reset_index(drop=True).copy(deep=True)

        # Instantiate `self.real_time_update_db`
        self._instantiate_real_time_update_db(db_index=self.records_db_images.index)

        self._pull_images_engine(save_dicom=save_dicom, allowed_modalities=allowed_modalities,
                                 save_png=save_png)
        self.real_time_update_db = self.real_time_update_db.replace({None: np.NaN})

        return _record_update_dbs_joiner(records_db=self.records_db_images, update_db=self.real_time_update_db)


# ----------------------------------------------------------------------------------------------------------
# Construct Database
# ----------------------------------------------------------------------------------------------------------


class CancerImageInterface(object):
    """

    Python Interface for the `Cancer Imaging Archive <http://www.cancerimagingarchive.net/>`_'s API.

    :param api_key: a key to the the Cancer Imaging Archive's API.

        .. note::

            An API key can be obtained by following the instructions provided `here <https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+%28REST+API%29+Usage+Guide/>`_.

    :type api_key: ``str``
    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    :param verbose: if ``True`` print additional details
    :type verbose: ``bool``
    
    .. warning::
    
        If you wish to use any data obtained from this resource for any form of publication,
        you must follow the citation guidelines provided by the study's authors on the project's
        Cancer Imaging Archive repository page.
    
    .. warning::
    
        Several studies on the Cancer Imaging Archive are subject to publication blockades.
        Therefore, if you intend to publish any findings which use data from this resource, you
        must first check that the studies you have selected are not subject to such restrictions.
    
    """

    def _save_cache_records_db(self):
        """

        Save ``cache_records_db`` to 'disk'.

        """
        self.cache_records_db.to_pickle(self._cache_records_db_save_path)

    def _load_prune_cache_records_db(self, load):
        """

        Load and Prune the ``cache_records_db``.

        :param load: if ``True`` load the ``cache_records_db`` dataframe in from disk.
        :type load: ``bool``
        """
        cache_records_db = pd.read_pickle(self._cache_records_db_save_path) if load else self.cache_records_db
        self.cache_records_db = _prune_rows_with_deleted_images(cache_records_db=cache_records_db,
                                                                columns=['cached_images_path',
                                                                         'cached_dicom_images_path'],
                                                                save_path=self._cache_records_db_save_path)

        # Recompute the image_count_converted_cache column following the pruning procedure.
        self.cache_records_db['image_count_converted_cache'] = self.cache_records_db['cached_images_path'].map(
            lambda x: len(x) if isinstance(x, (list, tuple)) else np.NaN)

    def _latent_temp_dir(self):
        """

        Merge any latent elements in the '__temp__' folder, if such a folder exists (and if it is populated).

        """
        if os.path.isdir(self._Images.temp_directory_path):
            temp_dir = self._Images.temp_directory_path
            latent_pickles = [os.path.join(temp_dir, i) for i in os.listdir(temp_dir) if i.endswith(".p")]
            if len(latent_pickles):
                with open(latent_pickles[0], "rb") as f:
                    settings_dict = pickle.load(f)
                settings_dict_for_pull = {k: v for k, v in settings_dict.items() if k not in ['records_db']}
                settings_dict_for_pull['new_records_pull'] = False  # adding this separately in `pull` breaks in python2

                print("\nResuming Download...")
                self.load_records_db(records_db=settings_dict['records_db'])
                self.pull(**settings_dict_for_pull)

    def _tcia_cache_records_db_handler(self):
        """

        Generate the `cache_records_db` database.
        If it does not exist, use `records_db`.
        If if already exists, merge it with `records_db_update`.

        """
        columns_to_drop = ['allowed_modality']

        def rows_to_conserve_func(x):
            """Mark to conserve the row in the cache if the conversion was successful or DICOMs were saved."""
            iccc, cdip = x['image_count_converted_cache'], x['cached_dicom_images_path']
            any_conversion_success = isinstance(iccc, (int, float)) and iccc > 0
            raw_dicoms = isinstance(cdip, (list, tuple)) and len(cdip) > 0  # > 0 needed b/c `True and 37 == 37`.
            return any_conversion_success or raw_dicoms

        def tcia_duplicates_handler(data_frame):
            """ Drop duplicate rows s.t. rows which reference 
            the most number of images are prefered."""
            # Cache_score is more important than keeping the most recent pull.
            sortby = ['biovida_version', '__image_cache_score__', 'pull_time']
            duplicates = ['series_instance_uid', 'study_instance_uid', 'query']

            def image_cache_score(row):
                def images_len(i):
                    return len(i) if isinstance(i, (list, tuple)) else 0
                return images_len(row['cached_images_path']) + images_len(row['cached_dicom_images_path'])

            data_frame['__image_cache_score__'] = data_frame.apply(image_cache_score, axis=1)
            data_frame = data_frame.sort_values(by=sortby)
            data_frame = data_frame.drop_duplicates(subset=duplicates, keep='last')
            return data_frame.drop('__image_cache_score__', axis=1)

        # Compose or update the master 'cache_records_db' dataframe
        if self.cache_records_db is None:
            cache_records_db = self.records_db.copy(deep=True)
            self.cache_records_db = cache_records_db[
                cache_records_db.apply(rows_to_conserve_func, axis=1)
            ].drop(columns_to_drop, axis=1).reset_index(drop=True)
            self._save_cache_records_db()
        else:
            columns_with_iterables_to_sort = ('cached_images_path', 'cached_dicom_images_path')
            self.cache_records_db = _records_db_merge(interface_name='CancerImageInterface',
                                                      current_records_db=self.cache_records_db,
                                                      records_db_update=self.records_db.drop(columns_to_drop, axis=1),
                                                      columns_with_dicts=('query',),
                                                      duplicates=tcia_duplicates_handler,
                                                      rows_to_conserve_func=rows_to_conserve_func,
                                                      columns_with_iterables_to_sort=columns_with_iterables_to_sort)

            # Save to disk
            self._save_cache_records_db()

        # Delete the '__temp__' folder
        shutil.rmtree(self._Images.temp_directory_path, ignore_errors=True)

    def __init__(self, api_key, cache_path=None, verbose=True):
        if os.path.isdir(api_key):
            raise ValueError("`api_key` must be not be a system path.")
        self._API_KEY = api_key
        self._cache_path = cache_path
        self._verbose = verbose
        self.dicom_modality_abbrevs = CancerImageArchiveParams(cache_path, verbose).dicom_modality_abbreviations('dict')

        # Root URL to for the Cancer Imaging Archive's REST API
        root_url = 'https://services.cancerimagingarchive.net/services/v3/TCIA'

        # Instantiate Classes
        self._Overview = _CancerImageArchiveOverview(dicom_modality_abbrevs=self.dicom_modality_abbrevs,
                                                     verbose=verbose,
                                                     cache_path=cache_path)

        self._tcia_overview_df = self._Overview._obtain_tcia_overview()

        self._Records = _CancerImageArchiveRecords(api_key=api_key,
                                                   dicom_modality_abbrevs=self.dicom_modality_abbrevs,
                                                   tcia_overview_df=deepcopy(self._tcia_overview_df),
                                                   root_url=root_url)

        self._Images = _CancerImageArchiveImages(api_key=api_key,
                                                 dicom_modality_abbrevs=self.dicom_modality_abbrevs,
                                                 root_url=root_url,
                                                 cache_path=cache_path,
                                                 verbose=verbose)

        self._ROOT_PATH = self._Overview._created_image_dirs['ROOT_PATH']

        # Search attributes
        self._pull_time = None
        self.search_dict = None
        self.current_query = None

        # Databases
        self.records_db = None
        self.pull_success = None

        # Path to the `cache_records_db`
        self._cache_records_db_save_path = os.path.join(self._Images._created_image_dirs['databases'],
                                                        'tcia_cache_records_db.p')

        # Load `cache_records_db` if it exists already, else set to None.
        if os.path.isfile(self._cache_records_db_save_path):
            self._load_prune_cache_records_db(load=True)
        else:
            self.cache_records_db = None

        # Load in databases in 'databases/__temp__', if they exist
        self._latent_temp_dir()

    def save_records_db(self, path):
        """

        Save the current ``records_db``.

        :param path: a system path ending with the '.p' file extension.
        :type path: ``str``
        """
        save_records_db(data_frame=self.records_db, path=path)

    def load_records_db(self, records_db):
        """

        Load a ``records_db``

        :param records_db: a system path or ``records_db`` itself.
        :type records_db: ``str`` or ``Pandas DataFrame``
        """
        if isinstance(records_db, pd.DataFrame):
            self.records_db = records_db
        else:
            self.records_db = pd.read_pickle(records_db)
        self._pull_time = self.records_db['pull_time'].iloc[0]
        last_query = self.records_db['query'].iloc[0]
        last_query['download_override'] = False
        last_query['pretty_print'] = False
        self.search(**last_query)

    @property
    def records_db_short(self):
        """Return `records_db` with nonessential columns removed."""
        return data_frame_col_drop(self.records_db, nonessential_cancer_image_columns, 'records_db')

    @property
    def cache_records_db_short(self):
        """Return `cache_records_db` with nonessential columns removed."""
        return data_frame_col_drop(self.cache_records_db, nonessential_cancer_image_columns, 'cache_records_db')
    
    def update_collections(self):
        """Refresh the list of collections provided by the Cancer Imaging Archive."""
        self._tcia_overview_df = self._Overview._obtain_tcia_overview(download_override=True)
        self._Records._tcia_overview_df = deepcopy(self._tcia_overview_df)

    @staticmethod
    def _collection_filter(summary_df, collection, cancer_type, location):
        """

        Limits `summary_df` to individual collections.

        :param summary_df: the yield of ``_CancerImageArchiveOverview()._obtain_tcia_overview()``
        :type summary_df: ``Pandas DataFrame``
        :param collection: a collection (study), or iterable (e.g., list) of collections,
                           hosted by the Cancer Imaging Archive. Defaults to ``None``.
        :type collection: ``list``, ``tuple``, ``str`` or ``None``
        :param cancer_type: a string or list/tuple of specifying cancer types. Defaults to ``None``.
        :type cancer_type: ``str``, ``iterable`` or ``None``
        :param location: a string or list/tuple of specifying body locations. Defaults to ``None``.
        :type location: ``str``, ``iterable`` or ``None``
        :return: ``summary_df`` limited to individual collections.
        :rtype: ``None`` or ``Pandas DataFrame``
        """
        # Filter by `collection`
        if isinstance(collection, (str, list, tuple)) and any(i is not None for i in (cancer_type, location)):
            raise ValueError("Both `cancer_types` and `location` must be ``None`` if a `collection` name is passed.")
        elif isinstance(collection, (str, list, tuple)):
            coll = [collection] if isinstance(collection, str) else collection
            summary_df = summary_df[summary_df['collection'].str.lower().isin(map(lambda x: cln(x).lower(), coll))]
            if summary_df.shape[0] == 0:
                raise AttributeError("No collection with the name '{0}' could be found.".format(collection))
            else:
                return summary_df.reset_index(drop=True)
        elif collection is not None:
            raise TypeError("'{0}' is an invalid type for `collection`.".format(str(type(collection).__name__)))

    def _search_dict_gen(self, collection, cancer_type, location, modality):
        """

        Generate a dictionary which contains the search information provided by the user.

        :param collection: See: ``search()``
        :type collection: ``str``, ``iterable`` or ``None``
        :param cancer_type: See: ``search()``
        :type cancer_type: ``str``, ``iterable`` or ``None``
        :param location: See: ``search()``
        :type location: ``str``, ``iterable`` or ``None``
        :param modality: See: ``search()``
        :type modality: ``str``, ``iterable`` or ``None``
        """
        # Note: lowered here because this is the behavior upstream.
        sdict = {'collection': collection, 'cancer_type': cancer_type, 'location': location, 'modality': modality}

        def lower_sdict(v):
            if isinstance(v, str):
                return v.lower()
            elif isinstance(v, (list, tuple)):
                return tuple(map(lambda x: x.lower(), v))
            else:
                return v

        self.search_dict = {k: lower_sdict(v) for k, v in sdict.items()}

    def search(self,
               collection=None,
               cancer_type=None,
               location=None,
               modality=None,
               download_override=False,
               pretty_print=True):
        """

        Method to Search for studies on the Cancer Imaging Archive.

        :param collection: a collection (study), or iterable (e.g., list) of collections,
                           hosted by the Cancer Imaging Archive. Defaults to ``None``.
        :type collection: ``list``, ``tuple``, ``str`` or ``None``
        :param cancer_type: a string or list/tuple of specifying cancer types. Defaults to ``None``.
        :type cancer_type: ``str``, ``iterable`` or ``None``
        :param location: a string or list/tuple of specifying body locations. Defaults to ``None``.
        :type location: ``str``, ``iterable`` or ``None``
        :param modality: the type of imaging technology. See: ``CancerImageInterface().dicom_modality_abbrevs`` for
                         valid values. Defaults to ``None``.
        :type modality: ``str``, ``iterable`` or ``None``
        :param download_override: If ``True``, override any existing database currently cached and download a new one.
                                  Defaults to ``False``.
        :type download_override: ``bool``
        :param pretty_print: if ``True``, pretty print the search results. Defaults to ``True``.
        :type pretty_print: ``bool``
        :return: a dataframe containing the search results.
        :rtype: ``Pandas DataFrame``

        :Example:

        >>> CancerImageInterface(YOUR_API_KEY_HERE).search(cancer_type='carcinoma', location='head')
        ...
           collection                   cancer_type                          modalities         subjects    location
        0  TCGA-HNSC            Head and Neck Squamous Cell Carcinoma  CT, MR, PT                 164     [Head, Neck]
        1  QIN-HeadNeck         Head and Neck Carcinomas               PT, CT, SR, SEG, RWV       156     [Head, Neck]
              ...                          ...                                  ...               ...         ...

        """
        self._search_dict_gen(collection, cancer_type, location, modality)

        if download_override:
            self.update_collections()
        summary_df = deepcopy(self._tcia_overview_df)

        if collection is not None:
            summary_df = self._collection_filter(summary_df,
                                                 collection=collection,
                                                 cancer_type=cancer_type,
                                                 location=location)
        else:
            summary_df = self._Overview._studies_filter(summary_df,
                                                        cancer_type=cancer_type,
                                                        location=location,
                                                        modality=modality)
            if summary_df.shape[0] == 0:
                raise NoResultsFound("Try Broadening the Search Criteria.")

        self.current_query = summary_df.reset_index(drop=True)

        if pretty_print and not IN_NOTEBOOK:
            pandas_pprint(data=self.current_query, full_cols=True,
                          col_align='left', column_width_limit=750)

        # Warn the user if search criteria have not been applied.
        if all([collection is None, cancer_type is None, location is None, modality is None]):
            sleep(0.25)
            warn("\nSpecific search criteria have not been applied.\n"
                 "If `pull()` is called, *all* collections will be downloaded.\n"
                 "Such a request could yield several terabytes of data.\n"
                 "If you still wish to proceed, consider adjusting `pull()`'s\n"
                 "`patient_limit` and `session_limit` parameters.")

        if not pretty_print or IN_NOTEBOOK:
            return self.current_query

    def _pull_records(self, patient_limit, collections_limit):
        """

        Pull Records from the TCIA API.

        :param patient_limit: limit on the number of patients to extract.
                             Patient IDs are sorted prior to this limit being imposed.
                             If ``None``, no patient_limit will be imposed. Defaults to `3`.
        :type patient_limit: ``int`` or ``None``
        :param collections_limit: limit the number of collections to download. If ``None``, no limit will be applied.
        :type collections_limit: ``int`` or ``None``
        :return: a list of dataframes
        :rtype: ``list``
        """
        pull_success = list()
        if isinstance(collections_limit, int):
            all_collections = self.current_query['collection'][:collections_limit]
        else:
            all_collections = self.current_query['collection']

        # Loop through and download all of the studies
        record_frames = list()
        for collection in all_collections:
            try:
                record_frames.append(self._Records.records_pull(study=collection,
                                                                search_dict=self.search_dict,
                                                                pull_time=self._pull_time,
                                                                patient_limit=patient_limit,
                                                                verbose=self._verbose))
                pull_success.append((True, collection))
            except IndexError as e:
                warn("\nIndexError Encountered: {0}".format(e))
                pull_success.append((False, collection))

        return record_frames, pull_success

    def _records_db_gen(self, patient_limit, collections_limit):
        """
        
        Generate ``records_db`` by refining the output of ``_pull_records()``.
        
        :param patient_limit: see ``pull()``.
        :type patient_limit: ``int`` or ``None``
        :param collections_limit: see ``pull()``.
        :type collections_limit: ``int`` or ``None``
        :return: 
        """
        record_frames, self.pull_success = self._pull_records(patient_limit=patient_limit,
                                                              collections_limit=collections_limit)

        # Check for failures
        download_failures = [collection for (success, collection) in self.pull_success if success is False]

        if len(download_failures) == len(self.pull_success):
            raise IndexError("Data could not be harvested for any of the requested collections.")
        elif len(download_failures):
            warn("\nThe following collections failed to download:\n{0}".format(
                list_to_bulletpoints(download_failures)))

        # Combine all record frames
        records_db = pd.concat(record_frames, ignore_index=True)
        records_db['biovida_version'] = [__version_numeric__] * records_db.shape[0]

        return records_db

    def extract_dicom_data(self, database='records_db', make_hashable=False):
        """

        Extract data from all dicom files referenced in ``records_db`` or ``cache_records_db``.
        Note: this requires that ``save_dicom`` is ``True`` when ``pull()`` is called.

        :param database: the name of the database to use. Must be one of: 'records_db', 'cache_records_db'.
                         Defaults to 'records_db'.
        :type database: ``str``
        :param make_hashable: If ``True`` convert the data extracted to nested tuples.
                              If ``False`` generate nested dictionaries. Defaults to ``False``
        :type make_hashable: ``bool``
        :return: a series of the dicom data with dictionaries of the form ``{path: {DICOM Description: value, ...}, ...}``.
                 If ``make_hashable`` is ``True``, all dictionaries will be converted to ``tuples``.
        :rtype: ``Pandas Series``
        """
        # ToDo: remove need for ``save_dicom=True`` by calling upstream before the temporary dicom files are destroyed.
        if database == 'records_db':
            database_to_use = self.records_db
        elif database == 'cache_records_db':
            database_to_use = self.cache_records_db
        else:
            raise ValueError("`database` must be one of 'records_db', 'cache_records_db'.")

        if not isinstance(database_to_use, pd.DataFrame):
            raise TypeError('`{0}` is not a DataFrame.'.format(database))
        else:
            db = database_to_use.copy(deep=True)

        def dicom_apply(paths):
            """Extract dicom data."""
            if not isinstance(paths, (list, tuple)):
                return paths
            elif not len(paths):
                return paths
            else:
                if make_hashable:
                    return tuple({p: tuple(dicom_to_dict(dicom_file=p).items()) for p in paths}.items())
                else:
                    return {p: dicom_to_dict(dicom_file=p) for p in paths}

        return pd.Series([dicom_apply(p) for p in db['cached_dicom_images_path']], index=db.index)

    def pull(self,
             patient_limit=3,
             session_limit=1,
             collections_limit=None,
             allowed_modalities=None,
             save_dicom=True,
             save_png=False,
             new_records_pull=True):
        """

        Pull (i.e., download) the current search.

        Notes:

        - When ``save_png`` is ``True``, 3D DICOM images are saved as individual frames.

        - PNG file names in the cache adhere to the following format:

            ``[instance, pull_position]__[patient_id_[Last 10 Digits of SeriesInstanceUID]]__[Image Scale ('default')].png``

        - DICOM file names in the cache adhere to the following format:
        
            ``[instance, original_name_in_source_file]__[patient_id_[Last 10 Digits of SeriesInstanceUID]].dcm``
        
        where:

        .. hlist::
            :columns: 1

            * 'instance' denotes the image's position in the 3D image (if applicable and available)
            * 'pull_position' denotes the position of the image in the set returned for the given 'SeriesInstanceUID' by the Cancer Imaging Archive.

        :param patient_limit: limit on the number of patients to extract.
                             Patient IDs are sorted prior to this limit being imposed.
                             If ``None``, no patient_limit will be imposed. Defaults to `3`.
        :type patient_limit: ``int`` or ``None``
        :param session_limit: restrict image harvesting to the first ``n`` imaging sessions (days) for a given patient,
                              where ``n`` is the value passed to this parameter. If ``None``, no limit will be imposed.
                              Defaults to `1`.

                .. warning::

                        Several studies (collections) in the Cancer Imaging Archive database have multiple imaging sessions.
                        Latter sessions may be of patients following interventions, such as surgery, intended to
                        *eliminate* cancerous tissue. For this reason it cannot be assumed that images obtained from
                        non-baseline sessions (i.e., session number > 1) contain signs of disease.

        :type session_limit: ``int``
        :param collections_limit: limit the number of collections to download. If ``None``, no limit will be applied.
                                  Defaults to ``None``.
        :type collections_limit: ``int`` or ``None``
        :param allowed_modalities: limit images downloaded to certain modalities.
                                   See: ``CancerImageInterface(YOUR_API_KEY_HERE).dicom_modality_abbrevs`` (use the keys).
                                   Note: 'MRI', 'PET', 'CT' and 'X-Ray' can also be used.
                                   This parameter is not case sensitive. Defaults to ``None``.
        :type allowed_modalities: ``list`` or ``tuple``
        :param save_dicom: if ``True``, save the DICOM images provided by The Cancer Imaging Archive 'as is'. Defaults to ``True``.
        :type save_dicom: ``bool``
        :param save_png: if ``True``, convert the DICOM images provided by The Cancer Imaging Archive to PNGs. Defaults to ``False``.
        :type save_png: ``bool``
        :param new_records_pull: if ``True``, download the data for the current search. If ``False``, use ``INSTANCE.records_db``.
        :type new_records_pull: ``bool``
        :return: a DataFrame with the record information.
        :rtype: ``Pandas DataFrame``
        """
        if new_records_pull:
            if not isinstance(self.current_query, pd.DataFrame):
                raise ValueError("`search()` must be called before `pull()`.")
            self._pull_time = datetime.now()
            self.records_db = self._records_db_gen(patient_limit=patient_limit,
                                                   collections_limit=collections_limit)
        elif not isinstance(self.records_db, pd.DataFrame):
            raise TypeError("`records_db` is not a DataFrame.")

        if save_png or save_dicom:
            self.records_db = self._Images.pull_images(records_db=self.records_db,
                                                       session_limit=session_limit,
                                                       save_png=save_png,
                                                       save_dicom=save_dicom,
                                                       allowed_modalities=allowed_modalities)

            # Add the new records_db datafame with the existing `cache_records_db`.
            self._tcia_cache_records_db_handler()

        return self.records_db
