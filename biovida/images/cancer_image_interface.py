"""

    Cancer Imaging Archive Interface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import io
import os
import dicom
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from time import sleep
from warnings import warn
from datetime import datetime

from biovida import __version__

# General Image Support Tools
from biovida.images._image_tools import TIME_FORMAT
from biovida.images._image_tools import NoResultsFound

# Database Management
from biovida.images._image_database_mgmt import load_temp_dbs
from biovida.images._image_database_mgmt import records_db_merge
from biovida.images._image_database_mgmt import record_update_dbs_joiner
from biovida.images._image_database_mgmt import prune_rows_with_deleted_images

# General Support Tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import only_numeric
from biovida.support_tools.support_tools import combine_dicts
from biovida.support_tools.support_tools import camel_to_snake_case
from biovida.support_tools.support_tools import list_to_bulletpoints

# Import Printing Tools
from biovida.support_tools.printing import pandas_pprint

# Cache Management
from biovida.support_tools._cache_management import package_cache_creator

# Cancer Image Support tools
from biovida.images._interface_support.dicom_data_to_dict import dicom_to_dict
from biovida.images._interface_support.cancer_image.cancer_image_parameters import CancerImgArchiveParams

# Spin up tqdm
tqdm.pandas("status")


# ----------------------------------------------------------------------------------------------------------
# Summarize Studies Provided Through the Cancer Imaging Archive
# ----------------------------------------------------------------------------------------------------------


class _CancerImgArchiveOverview(object):
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
        _, self._created_img_dirs = package_cache_creator(sub_dir='images', cache_path=cache_path,
                                                          to_create=['tcia'], nest=[('tcia', 'databases')])

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

        # Drop Studies which are 'Coming Soon'.
        summary_df = summary_df[summary_df['status'].str.strip().str.lower() != 'coming soon']

        # Drop Studies which are on phantoms
        summary_df = summary_df[~summary_df['location'].str.lower().str.contains('phantom')]

        # Drop Studies which are on mice or phantoms
        summary_df = summary_df[~summary_df['collection'].str.lower().str.contains('mouse|phantom')]

        # Only Keep Studies which are public
        summary_df = summary_df[summary_df['access'].str.strip().str.lower() == 'public'].reset_index(drop=True)

        # Add Full Name for Modalities
        summary_df['modalities_full'] = summary_df['modalities'].map(
            lambda x: [self.dicom_modality_abbrevs.get(cln(i), i) for i in cln(x).split(", ")])

        # Parse the Location Column (and account for special case: 'Head-Neck').
        summary_df['location'] = summary_df['location'].map(
            lambda x: cln(x.replace(" and ", ", ").replace("Head-Neck", "Head, Neck")).split(", "))

        # Convert 'Update' to Datetime
        summary_df['updated'] = pd.to_datetime(summary_df['updated'], infer_datetime_format=True)

        # Clean Column names
        summary_df.columns = list(map(lambda x: cln(x, extent=2), summary_df.columns))

        return summary_df

    def _all_studies_cache_mngt(self, download_override):
        """

        Obtain and Manage a copy the table which summarizes the the Cancer Imaging Archive
        on the organization's homepage.

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
        :type download_override: ``bool``
        :return: summary table hosted on the home page of the Cancer Imaging Archive.
        :rtype: ``Pandas DataFrame``
        """
        # Define the path to save the data
        save_path = os.path.join(self._created_img_dirs['databases'], 'all_tcia_studies.p')

        if not os.path.isfile(save_path) or download_override:
            if self._verbose:
                header("Downloading Table of Available Studies... ", flank=False)
            summary_df = self._all_studies_parser()
            summary_df.to_pickle(save_path)
        else:
            summary_df = pd.read_pickle(save_path)

        return summary_df

    def _studies_filter(self, summary_df, cancer_type, location, modality):
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


class _CancerImgArchiveRecords(object):
    """

    Class to harvest records for a given collection/study through the Cancer Imaging Archive API.

    :param api_key: an key to the the Cancer Imaging Archive's API.
    :type api_key: ``str``
    :param dicom_modality_abbrevs: an instance of ``CancerImgArchiveParams().dicom_modality_abbreviations('dict')``
    :type dicom_modality_abbrevs: ``dict``
    :param cancer_img_archive_overview: an instance of ``_CancerImgArchiveOverview()``
    :type cancer_img_archive_overview: ``class``
    :param root_url: the root URL for the Cancer Imaging Archive's API.
    :type root_url: ``str``
    """

    def __init__(self, api_key, dicom_modality_abbrevs, cancer_img_archive_overview, root_url):
        self.ROOT_URL = root_url
        self.records_df = None
        self.dicom_modality_abbrevs = dicom_modality_abbrevs
        self._Overview = cancer_img_archive_overview
        self.API_KEY = api_key
        self._url_sep = '+'

    def _study_extract(self, study):
        """

        Download all patients in a given study.

        :param study: a Cancer Imaging Archive collection (study)
        :type study: ``str``
        :return: the yeild of passing the ``getPatientStudy`` parameter to the Cancer Imaging Archive API for a given
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

        :raises IndexError: if both '+' and '-' fail to yeild a dataframe with nonzero length.
        """
        study_df = self._study_extract(study)
        if study_df.shape[0] == 0:
            self._url_sep = '-'
            study_df = self._study_extract(study)
            self._url_sep = '+'  # reset
            if study_df.shape[0] == 0:
                raise IndexError("The '{0}' collection/study data has no length.\n"
                                 "The separator being used to replace spaces in the URL may be incorrect. An attempt\n"
                                 "was made with the following separators: '+' and '-'. Alternatively, this problem\n"
                                 "could be caused by a problem with the Cancer Imaging Archive API.\n".format(study))
        return study_df

    def _date_index_map(self, list_of_dates):
        """

        Returns a dict of the form: ``{date: index in ``list_of_dates``, ...}``

        :param list_of_dates: a list (or tuple) of datetime objects.
        :type list_of_dates: ``list`` or ``tuple``
        :return: descrition (above)
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

        # Convert study_date to datetime
        study_df['study_date'] = pd.to_datetime(study_df['study_date'], infer_datetime_format=True)

        # Divide Study into stages (e.g., Baseline (session 1); Baseline + 1 Month (session 2), etc.
        stages = study_df.groupby('patient_id').apply(lambda x: self._date_index_map(x['study_date'].tolist())).to_dict()

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

    def _patient_img_summary(self, patient, study, patient_dict):
        """

        Harvests the Cancer Image Archive's Text Record of all baseline images for a given patient
        in a given study.

        :param patient: the patient_id (will be used to form the request to the TCIA server).
        :type patient: ``str``
        :param study: a Cancer Imaging Archive collection (study).
        :type study: ``str``
        :param patient_dict: a value in ``study_dict`` (which is a dictionary itself).
        :type patient_dict: ``dict``
        :return: the yeild of the TCIA ``getSeries`` param for a given patient in a given collection (study).
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

        # Add sex, age, session, and study_date
        patient_info = patient_df['study_instance_uid'].map(
            lambda x: {k: patient_dict[x][k] for k in ('sex', 'age', 'session', 'study_date')})
        patient_df = patient_df.join(pd.DataFrame(patient_info.tolist()))

        # Add patient_id
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
            lambda x: {'F': 'female', 'M': 'male'}.get(cln(str(x)).upper(), x), na_action='ignore')

        # Convert entries in the 'age' Column to floats.
        patient_study_df['age'] = patient_study_df['age'].map(
            lambda x: only_numeric(x) / 12.0 if 'M' in str(x).upper() else only_numeric(x), na_action='ignore')

        # Remove unneeded line break marker
        for c in ('protocol_name', 'series_description'):
            patient_study_df[c] = patient_study_df[c].map(lambda x: cln(x.replace("\/", " ")), na_action='ignore')

        # Add the full name for modality.
        patient_study_df['modality_full'] = patient_study_df['modality'].map(
            lambda x: self.dicom_modality_abbrevs.get(x, np.NaN), na_action='ignore')

        # Convert series_date to datetime
        patient_study_df['series_date'] = pd.to_datetime(patient_study_df['series_date'], infer_datetime_format=True)

        # Sort and Return
        return patient_study_df.sort_values(by=['patient_id', 'session']).reset_index(drop=True)

    def _get_condition_name(self, collection_series, overview_download_override):
        """
        
        This method gets the name of the condition studied for a given collection (study).
        (collection_series gets reduced down to a single unique).

        :param collection_series: a series of the study name, e.g., ('MY-STUDY', 'MY-STUDY', 'MY-STUDY', ...).
        :type collection_series: ``Pandas Series``
        :param overview_download_override: see ``_CancerImgArchiveOverview()_all_studies_cache_mngt()``'s
                                           ``download_override`` param.
        :type overview_download_override: ``bool``
        :return: the name of disease studied in a given collection (lower case).
        :rtype: ``str``
        """
        unique_studies = collection_series.unique()
        if len(unique_studies) == 1:
            collection = unique_studies[0]
        else:
            raise AttributeError("`{0}` studies found in `records`. Expected one.".format(str(len(unique_studies))))
        summary_df = self._Overview._all_studies_cache_mngt(download_override=overview_download_override)
        condition_name = summary_df[summary_df['collection'] == collection]['cancer_type'].iloc[0]
        return condition_name.lower() if isinstance(condition_name, str) else condition_name

    def records_pull(self, study, search_dict, pull_time, overview_download_override=False, patient_limit=3):
        """

        Extract record of all images for all patients in a given study.

        :param study: a Cancer Imaging Archive collection (study).
        :type study: ``str``
        :param search_dict: a dicitionary which contains the search information provided by the user
                            (as evolved inside  ``CancerImageInterface()_search_dict_gen()``.
        :type search_dict: ``dict``
        :param pull_time: the time the query was launched.
        :type pull_time: ``datetime``
        :param overview_download_override: see ``_CancerImgArchiveOverview()_all_studies_cache_mngt()``'s
                                           ``download_override`` param.
        :type overview_download_override: ``bool``
        :param patient_limit: limit on the number of patients to extract.
                             Patient IDs are sorted prior to this limit being imposed.
                             If ``None``, no `patient_limit` will be imposed. Defaults to `3`.
        :type patient_limit: ``int`` or ``None``
        :return: a dataframe of all baseline images
        :rtype: ``Pandas DataFrame``
        """
        # Summarize a study by patient
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
        for patient in tqdm(patients_to_obtain):
            frames.append(self._patient_img_summary(patient, study=study, patient_dict=study_dict[patient]))

        # Concatenate baselines frame for each patient
        patient_study_df = pd.concat(frames, ignore_index=True)

        # Add Study name
        patient_study_df['study_name'] = study

        # Add the Name of the illness
        patient_study_df['cancer_type'] = self._get_condition_name(patient_study_df['collection'],
                                                                   overview_download_override)

        # Add the Search query which created the current results and the time the search was launched.
        patient_study_df['query'] = [search_dict] * patient_study_df.shape[0]
        patient_study_df['pull_time'] = [pull_time] * patient_study_df.shape[0]

        # Clean the dataframe
        self.records_df = self._clean_patient_study_df(patient_study_df)

        return self.records_df


# ----------------------------------------------------------------------------------------------------------
# Pull Images from the Cancer Imaging Archive
# ----------------------------------------------------------------------------------------------------------


class _CancerImgArchiveImages(object):
    """

    Class to harvest images for a given collection/study through the Cancer Imaging Archive API, based on
    records extracted by ``_CancerImgArchiveRecords()``.

    :param api_key: an key to the the Cancer Imaging Archive's API.
    :type api_key: ``str``
    :param dicom_modality_abbrevs: an instance of ``CancerImgArchiveParams().dicom_modality_abbreviations('dict')``
    :type dicom_modality_abbrevs: ``dict``
    :param root_url: the root URL for the the Cancer Imaging Archive's API.
    :type root_url: ``str``
    :param verbose: print additional details.
    :type verbose: ``bool``
    :type cache_path: ``str`` or ``None``
    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    """

    def __init__(self, api_key, dicom_modality_abbrevs, root_url, verbose, cache_path=None):
        _, self._created_img_dirs = package_cache_creator(sub_dir='images',
                                                          cache_path=cache_path,
                                                          to_create=['tcia'],
                                                          nest=[('tcia', 'raw'), ('tcia', 'dicoms'), ('tcia', 'databases')],
                                                          verbose=verbose)

        self.API_KEY = api_key
        self.dicom_modality_abbrevs = dicom_modality_abbrevs
        self.ROOT_URL = root_url

        # Add Record DataFrame; this database updates in real time as the images are downloaded.
        self.records_db_images = None
        self.real_time_update_db = None
        self.real_time_update_db_path = None

        # Define the path to the temporary directory
        self.temp_directory_path = os.path.join(self._created_img_dirs['databases'], "__temp__")

    def _create_temp_directory(self):
        """

        Create the temporary folder for ``real_time_update_db``.

        """
        if not os.path.isdir(self.temp_directory_path):
            os.makedirs(self.temp_directory_path)

    def _instantiate_real_time_update_db(self, db_index, pull_time):
        """

        Create the ``real_time_update_db`` and define the path to the location where it will be saved.

        :param db_index: the index of the ``real_time_update_db`` dataframe (should be from ``records_db``).
        :type db_index: ``Pandas Series``
        :param pull_time: see ``pull_images()``
        :type pull_time: ``str``
        """
        # Define the path to save `self.real_time_update_db` to.
        self.real_time_update_db_path = os.path.join(self.temp_directory_path, "{0}__update_db.p".format(pull_time))

        # Define columns
        real_time_update_columns = ['raw_dicom_files_paths', 'cached_images_path', 'conversion_success',
                                    'allowed_modality', 'image_count_converted_cache']

        # Instantiate
        db = pd.DataFrame(columns=real_time_update_columns, index=db_index).replace({np.NaN: None})
        self.real_time_update_db = db
        
    def _save_real_time_update_db(self):
        """

        Save the ``real_time_update_db`` to disk.

        """
        # Save the `real_time_update_db` to disk.
        self.real_time_update_db.to_pickle(self.real_time_update_db_path)

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

    def _dicom_to_standard_image(self, f, pull_position, conversion, new_file_name, image_format):
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
        :param conversion: the color scale conversion to use, e.g., 'LA' or 'RGB'.
        :type conversion: ``str``
        :param new_file_name: see ``_save_dicom_as_img``'s ``save_name`` parameter.
        :type new_file_name: ``str``
        :param image_format: see: ``pull_images()``.
        :type image_format: ``str``
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

        save_location = self._created_img_dirs['raw']

        def save_path(instance):
            """Define the path to save the image to."""
            head = "{0}_{1}".format(instance, pull_position)
            file_name = "{0}__{1}__default.{2}".format(head, os.path.basename(new_file_name), image_format.replace(".", ""))
            return os.path.join(save_location, file_name)

        if pixel_arr.ndim == 2:
            # Define save name by combining the images instance in the set, `new_file_name` and `image_format`.
            instance = cln(str(f.InstanceNumber)) if len(cln(str(f.InstanceNumber))) else '0'
            path = save_path(instance)
            Image.fromarray(pixel_arr).convert(conversion).save(path)
            all_save_paths.append(path)
        # If ``f`` is a 3D image (e.g., segmentation dicom files), save each layer as a seperate file/image.
        elif pixel_arr.ndim == 3:
            for instance, layer in enumerate(range(pixel_arr.shape[0]), start=1):
                path = save_path(instance)
                Image.fromarray(pixel_arr[layer:layer + 1][0]).convert(conversion).save(path)
                all_save_paths.append(path)
        else:
            raise ValueError("Cannot handle {0} dimensional arrays. Images must be 2D or 3D.".format(pixel_arr.ndim))

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
        # Extract the current value.
        current = self.real_time_update_db.get_value(index, column)

        def cleaner(to_clean):
            """Tool which will ensure the output is a `list`."""
            return list(to_clean) if isinstance(to_clean, (list, tuple)) else []

        # Clean `current` and `new`, combine and generate a replacement candidate
        replacement_candidate = tuple(cleaner(current) + cleaner(new))

        # Generate the replacement
        replacement = replacement_candidate if len(replacement_candidate) else np.NaN

        # Set the value
        self.real_time_update_db.set_value(index, column, replacement)

        # Return the length if requested.
        return len(replacement) if return_replacement_len and isinstance(replacement, (list, tuple)) else None

    def _save_dicom_as_img(self,
                           path_to_dicom_file,
                           index,
                           pull_position,
                           save_name=None,
                           color=False,
                           image_format='png'):
        """

        Save a dicom image as a more common file format.

        :param path_to_dicom_file: path to a dicom image
        :type path_to_dicom_file: ``str``
        :param pull_position: the position of the image in the raw zip data provided by the Cancer Imaging Archive API.
        :type pull_position: ``int``
        :param index: the row index currently being processed inside of the main loop in ``_pull_images_engine()``.
        :type index: ``int``
        :param save_name: name of the new file (do *NOT* include a file extension).
                          To specify a file format, use ``image_format``.
                          If ``None``, name from ``path_to_dicom_file`` will be conserved.
        :type save_name: ``str``
        :param color: If ``True``, convert the image to RGB before saving. If ``False``, save as a grayscale image.
                      Defaults to ``False``
        :type color: ``bool``
        :param image_format: see: ``pull_images()``.
        :type image_format: ``str``
        :return: ``_dicom_to_standard_image()``
        :rtype: ``tuple``
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
        all_save_paths, success = self._dicom_to_standard_image(f, pull_position, conversion, new_file_name, image_format)

        # Update Record
        cfp_len = self._update_and_set_list(index, 'cached_images_path', all_save_paths, return_replacement_len=True)
        self.real_time_update_db.set_value(index, 'image_count_converted_cache', cfp_len)

        # Add record of whether or not the dicom file could be converted to a standard image type
        self.real_time_update_db.set_value(index, 'conversion_success', success)

        # Save the data frame
        self._save_real_time_update_db()

    def _move_dicoms(self, save_dicoms, dicom_files, series_abbrev, index):
        """

        Move the dicom source files to ``self._created_img_dirs['dicoms']``.
        Employ to prevent the raw dicom files from being destroyed.

        :param save_dicoms: see: ``pull_images()``
        :type save_dicoms: ``bool``
        :param dicom_files: the yield of ``_download_zip()``
        :type dicom_files: ``list``
        :param series_abbrev: as evolved inside ``_pull_images_engine()``.
        :type series_abbrev: ``str``
        :param index: the row index currently being processed inside of the main loop in ``_pull_images_engine()``.
        :type index: ``int``
        """
        if save_dicoms is not True:
            self.real_time_update_db.set_value(index, 'raw_dicom_files_paths', np.NaN)
            self._save_real_time_update_db()
            return None

        new_dircom_paths = list()
        for f in dicom_files:
            # Define a name for the new file by extracting the dicom file name and combining with `series_abbrev`.
            f_parsed = list(os.path.splitext(os.path.basename(f)))
            new_dicom_file_name = "{0}__{1}{2}".format(f_parsed[0], series_abbrev, f_parsed[1])

            # Define the location of the new files
            new_location = os.path.join(self._created_img_dirs['dicoms'], new_dicom_file_name)
            new_dircom_paths.append(new_location)

            # Move the dicom file from '__temp__' --> to --> new location
            os.rename(f, new_location)

        # Update the save dataframe
        self.real_time_update_db.set_value(index, 'raw_dicom_files_paths', tuple(new_dircom_paths))

        # Save the data frame.  Note: if python crashes or the session ends before the above loop completes,
        # information on the partial transfer will be lost.
        self._save_real_time_update_db()

    def _cache_check(self, check_cache_first, series_abbrev, n_images_min, save_dicoms):
        """

        Check that caches likely contain that data which would be obtained by downloading it from the database.

        :param series_abbrev: as evolved inside ``_pull_images_engine()``
        :type series_abbrev: ``str``
        :param n_images_min: `image_count` as passed in ``_pull_images_engine()``. Denotes the min. number of images
                              for a given series for the cache to be considered complete (less than this number
                              will trigger an effort to download the corresponding images (or, more specifically,
                              seriesUID).
        :type n_images_min: ``int``
        :return: tuple of the form:

        ``(cache likely complete,
           series_abbrev matches in self._created_img_dirs['raw'],
           series_abbrev matches in self._created_img_dirs['dicoms'])``

        :rtype: ``tuple``
        """
        # Instruct ``_pull_images_engine()`` to download the images without checking the cache first.
        if check_cache_first is False:
            return False, None, None

        # Check that `self._created_img_dirs['raw']` has files which contain the string `series_abbrev`.
        save_location_summary = [os.path.join(self._created_img_dirs['raw'], f)
                                 for f in os.listdir(self._created_img_dirs['raw']) if series_abbrev in f]

        # Check that `self._created_img_dirs['dicoms'])` has files which contain the string `series_abbrev`.
        dicoms_sl_summary = tuple([os.path.join(self._created_img_dirs['dicoms'], f)
                                   for f in os.listdir(self._created_img_dirs['dicoms']) if series_abbrev in f])

        # Base determination of whether or not the cache is complete w.r.t. dicoms on `save_dicoms`.
        dicoms_sl_summary_complete = len(dicoms_sl_summary) >= n_images_min if save_dicoms else True

        # Compose completeness boolean from the status of `self._created_img_dirs['raw']` and
        # `self._created_img_dirs['dicoms']`
        complete = len(save_location_summary) >= n_images_min and dicoms_sl_summary_complete

        return complete, save_location_summary, dicoms_sl_summary if len(dicoms_sl_summary) else np.NaN

    def _create_temp_dir(self, temp_folder_name='__temp__'):
        """

        Create a temporary directory.

        :param temp_folder_name: name for the temporary directory to be created.
        :type temp_folder_name: ``str``
        :return: the full path to the newly created temporary directory.
        :rtype: ``str``
        """
        temp_folder = os.path.join(self._created_img_dirs['dicoms'], temp_folder_name)
        if os.path.isdir(temp_folder):
            shutil.rmtree(temp_folder, ignore_errors=True)
        os.makedirs(temp_folder)
        return temp_folder

    def _valid_modality(self, allowed_modalities, modality, modality_full):
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

    def _pull_images_engine(self, save_dicoms, allowed_modalities, image_format, check_cache_first):
        """

        Tool to coordinate the above machinery for pulling and downloading images (or locating them in the cache).

        :param save_dicoms: see: ``pull_images()``.
        :type save_dicoms: ``bool``
        :param image_format: see: ``pull_images()``
        :param image_format: ``str``
        :param allowed_modalities: see: ``pull_images()``
        :type allowed_modalities: ``list``, ``tuple`` or ``None``.
        :param check_cache_first: see: ``pull_images()``
        :param check_cache_first: ``bool``
        """
        columns = ('series_instance_uid', 'patient_id', 'image_count', 'modality', 'modality_full')
        zipped_cols = list(zip(*[self.records_db_images[c] for c in columns] + [pd.Series(self.records_db_images.index)]))

        for series_uid, patient_id, image_count, modality, modality_full, index in tqdm(zipped_cols):
            # Check if the image should be harvested (or loaded from the cache).
            valid_image = self._valid_modality(allowed_modalities, modality, modality_full)

            # Add whether or not the image was of the modality (or modalities) requested by the user.
            self.real_time_update_db.set_value(index, 'allowed_modality', valid_image)

            # Compose central part of the file name from 'patient_id' and the last ten digits of 'series_instance_uid'
            series_abbrev = "{0}_{1}".format(patient_id, str(series_uid)[-10:])

            # Analyze the cache to determine whether or not downloading the images is needed
            cache_complete, sl_summary, dsl_summary = self._cache_check(check_cache_first=check_cache_first,
                                                                        series_abbrev=series_abbrev,
                                                                        n_images_min=image_count,
                                                                        save_dicoms=save_dicoms)

            if valid_image and not cache_complete:
                temporary_folder = self._create_temp_dir()

                # Download the images into a temporary folder.
                dicom_files = self._download_zip(series_uid, temporary_folder=temporary_folder)

                # Convert dicom files to `image_format`
                for e, f in enumerate(dicom_files, start=1):
                    self._save_dicom_as_img(f, index, pull_position=e, save_name=series_abbrev, image_format=image_format)

                # Save raw dicom files, if `save_dicoms` is True.
                self._move_dicoms(save_dicoms, dicom_files, series_abbrev, index)

                # Delete the temporary folder.
                shutil.rmtree(temporary_folder, ignore_errors=True)
            else:
                self._update_and_set_list(index, 'raw_dicom_files_paths', dsl_summary)
                self._update_and_set_list(index, 'cached_images_path', sl_summary)
                self.real_time_update_db.set_value(index, 'conversion_success', cache_complete)
                self.real_time_update_db.set_value(index, 'image_count_converted_cache', len(sl_summary))
                # Save the data frame
                self._save_real_time_update_db()

    def pull_images(self,
                    records_db,
                    pull_time,
                    session_limit=1,
                    image_format='png',
                    save_dicoms=True,
                    allowed_modalities=None,
                    check_cache_first=True):
        """

        Pull Images from the Cancer Imaging Archive.

        :param records_db: the yield from ``_CancerImgArchiveRecords().records_pull()``.
        :type records_db: ``Pandas DataFrame``
        :param pull_time: the time the pull for images was initiated (standard format: "%Y_%h_%d__%H_%M_%S_%f").
        :type pull_time: ``str``
        :param session_limit: restrict image harvesting to the first ``n`` sessions, where ``n`` is the value passed
                              to this parameter. If ``None``, no limit will be imposed. Defaults to `1`.
        :type session_limit: ``int``
        :param image_format: format for the image, e.g., 'png', 'jpg', etc. Defaults to 'png'.
        :type image_format: ``str``
        :param save_dicoms: if ``True``, save the raw dicom files. Defaults to ``False``.
        :type save_dicoms: ``bool``
        :param allowed_modalities: limit images downloaded to certain modalities.
                                   See: CancerImageInterface().dicom_modality_abbrevs (use the keys).
                                   Note: 'MRI', 'PET', 'CT' and 'X-Ray' can also be used.
                                   This parameter is not case sensitive. Defaults to ``None``.
        :type allowed_modalities: ``list``, ``tuple`` or ``None``.
        :param check_cache_first: check the image cache for the image prior to downloading.
                                  If the image is already present, no attempt will be made to download it again.
        :type check_cache_first: ``bool``
        :return: a dataframe with information about the images cached by this method.
        :rtype: ``Pandas DataFrame``
        """
        # Notes on 'image_count_converted_cache':
        # 1. a column which provides the number of images each SeriesInstanceUID yeilded
        # 2. values may be discrepant with the 'image_count' column because 3D images are expanded
        #    into their individual frames when saved to the converted images cache.

        # Create the __temp__ folder if it does not already exist.
        self._create_temp_directory()

        # Apply limit on number of sessions, if any
        if isinstance(session_limit, int):
            if session_limit < 1:
                raise ValueError("`session_limit` must be greater than or equal to 1.")
            self.records_db_images = records_db[records_db['session'].map(
                lambda x: float(x) <= session_limit if pd.notnull(x) else False)].reset_index(drop=True).copy(deep=True)
        else:
            self.records_db_images = records_db.reset_index(drop=True).copy(deep=True)

        # Save `records_db_images` to the __temp__ folder
        self.records_db_images.to_pickle(os.path.join(self.temp_directory_path, "{0}__records_db.p".format(pull_time)))

        # Instantiate `self.real_time_update_db`
        self._instantiate_real_time_update_db(db_index=self.records_db_images.index, pull_time=pull_time)

        # Harvest images
        self._pull_images_engine(save_dicoms, allowed_modalities, image_format, check_cache_first)
        self.real_time_update_db = self.real_time_update_db.replace({None: np.NaN})

        return record_update_dbs_joiner(records_db=self.records_db_images, update_db=self.real_time_update_db)


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
    :param verbose: print additional details.
    :type verbose: ``bool``
    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    """

    def _latent_temp_dir(self):
        """

        Merge any latent elements in the '__temp__' folder, if such a folder exists (and if it is populated).

        """
        if os.path.isdir(self._Images.temp_directory_path):
            latent_temps = load_temp_dbs(self._Images.temp_directory_path)
            if latent_temps is not None:
                self._tcia_cache_records_db_gen(tcia_cache_records_db_update=latent_temps)
            # Delete the latent '__temp__' folder
            shutil.rmtree(self._Images.temp_directory_path, ignore_errors=True)

    def _tcia_cache_records_db_gen(self, tcia_cache_records_db_update):
        """

        Generate the `cache_records_db` database.
        If it does not exist, use `cache_records_db`.
        If if already exists, merge it with `tcia_cache_records_db_update`.

        :param tcia_cache_records_db_update: the new search dataframe to added to the existing one.
        :type tcia_cache_records_db_update: ``Pandas DataFrame``
        """
        def rows_to_conserve_func(x):
            """Mark to conserve the row in the cache if the conversion was sucessful or dicoms were saved."""
            conversion_success = x['conversion_success'] == True
            raw_dicoms = isinstance(x['raw_dicom_files_paths'], (list, tuple)) and len(x['raw_dicom_files_paths'])
            return conversion_success or raw_dicoms

        # Compose or update the master 'cache_records_db' dataframe
        if self.cache_records_db is None:
            cache_records_db = tcia_cache_records_db_update.copy(deep=True)
            self.cache_records_db = cache_records_db[
                cache_records_db.apply(rows_to_conserve_func, axis=1)].reset_index(drop=True)
            self.cache_records_db.to_pickle(self._tcia_cache_records_db_save_path)
        else:
            duplicates_subset_columns = [c for c in self.cache_records_db.columns if c != 'pull_time']
            columns_with_iterables_to_sort = ('cached_images_path', 'raw_dicom_files_paths')
            self.cache_records_db = records_db_merge(current_records_db=self.cache_records_db,
                                                     records_db_update=tcia_cache_records_db_update,
                                                     columns_with_dicts=('query',),
                                                     duplicates_subset_columns=duplicates_subset_columns,
                                                     rows_to_conserve_func=rows_to_conserve_func,
                                                     columns_with_iterables_to_sort=columns_with_iterables_to_sort,
                                                     relationship_mapping_func=None)

            # Save to disk
            self.cache_records_db.to_pickle(self._tcia_cache_records_db_save_path)

    def __init__(self, api_key, verbose=True, cache_path=None):
        self._verbose = verbose
        self.dicom_modality_abbrevs = CancerImgArchiveParams(cache_path, verbose).dicom_modality_abbreviations('dict')

        # Root URL to for the Cancer Imaging Archive's REST API
        root_url = 'https://services.cancerimagingarchive.net/services/v3/TCIA'

        # Instantiate Classes
        self._Overview = _CancerImgArchiveOverview(dicom_modality_abbrevs=self.dicom_modality_abbrevs,
                                                   verbose=verbose,
                                                   cache_path=cache_path)

        self._Records = _CancerImgArchiveRecords(api_key=api_key,
                                                 dicom_modality_abbrevs=self.dicom_modality_abbrevs,
                                                 cancer_img_archive_overview=self._Overview,
                                                 root_url=root_url)

        self._Images = _CancerImgArchiveImages(api_key=api_key,
                                               dicom_modality_abbrevs=self.dicom_modality_abbrevs,
                                               root_url=root_url,
                                               cache_path=cache_path,
                                               verbose=verbose)

        # Search attributes
        self._pull_time = None
        self.search_dict = None
        self.current_query = None

        # Databases
        self.records_db = None

        # Path to the `cache_records_db`
        self._tcia_cache_records_db_save_path = os.path.join(self._Images._created_img_dirs['databases'],
                                                            'tcia_cache_records_db.p')

        # Load `cache_records_db` if it exists already, else set to None.
        if os.path.isfile(self._tcia_cache_records_db_save_path):
            cache_records_db = pd.read_pickle(self._tcia_cache_records_db_save_path)
            self.cache_records_db = prune_rows_with_deleted_images(cache_records_db=cache_records_db,
                                                                   columns=['cached_images_path', 'raw_dicom_files_paths'],
                                                                   save_path=self._tcia_cache_records_db_save_path)

            # Recompute the image_count_converted_cache column following the pruning procedure.
            self.cache_records_db['image_count_converted_cache'] = self.cache_records_db['cached_images_path'].map(
                lambda x: len(x) if isinstance(x, (list, tuple)) else np.NaN)
        else:
            self.cache_records_db = None

        if os.path.isdir(self._Images.temp_directory_path):
            self._latent_temp_dir()

    def _collection_filter(self, summary_df, collection, cancer_type, location):
        """

        Limits `summary_df` to individual collections.

        :param summary_df: the yeild of ``_CancerImgArchiveOverview()._all_studies_cache_mngt()``
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
        # Create the search dict.
        self._search_dict_gen(collection, cancer_type, location, modality)

        # Load the Summary Table
        summary_df = self._Overview._all_studies_cache_mngt(download_override)

        # Filter by `collection`
        if collection is not None:
            summary_df = self._collection_filter(summary_df, collection, cancer_type, location)
        else:
            # Apply Filters
            summary_df = self._Overview._studies_filter(summary_df, cancer_type, location, modality)
            if summary_df.shape[0] == 0:
                raise NoResultsFound("Try Broadening the Search Criteria.")

        # Cache Search
        self.current_query = summary_df.reset_index(drop=True)

        if pretty_print:
            current_query_print = self.current_query.copy(deep=True)
            pandas_pprint(current_query_print, full_cols=True, col_align='left', lift_column_width_limit=True)

        # Warn the user if search criteria have not been applied.
        if all([collection is None, cancer_type is None, location is None, modality is None]):
            sleep(0.25)
            warn("\nSpecific search criteria have not been applied.\n"
                 "If `pull()` is called, *all* collections will be downloaded.\n"
                 "Such a request could yield several terabytes of data.\n"
                 "If you still wish to proceed, consider adjusting `pull()`'s\n"
                 "`patient_limit` and `session_limit` parameters.")

        return None if pretty_print else self.current_query

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
            if self._verbose:
                print("\nDownloading Records for the '{0}' Collection...".format(collection))
            try:
                record_frames.append(self._Records.records_pull(study=collection,
                                                                search_dict=self.search_dict,
                                                                pull_time=self._pull_time,
                                                                patient_limit=patient_limit))
                pull_success.append((True, collection))
            except IndexError as e:
                warn("\nIndexError Encountered: {0}".format(e))
                pull_success.append((False, collection))

        return record_frames, pull_success

    def _tcia_cache_records_db_handler(self):
        """

        If ``cache_records_db`` does not exists on disk, create it using ``tcia_cache_records_db_update`` (evolved inside this
        function). If it already exists, merge it with ``tcia_cache_records_db_update``.

        """
        # Generate the `cache_records_db`.
        self._tcia_cache_records_db_gen(tcia_cache_records_db_update=load_temp_dbs(self._Images.temp_directory_path))

        # Delete the '__temp__' folder
        shutil.rmtree(self._Images.temp_directory_path, ignore_errors=True)

    def extract_dicom_data(self, database='records_db', make_hashable=False):
        """

        Extract data from all dicom files referenced in ``records_db`` or ``cache_records_db``.
        Note: this requires that ``save_dicoms`` is ``True`` when ``pull()`` is called.

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
        # ToDo: remove need for ``save_dicoms=True`` by calling upstream before the temporary dicom files are destroyed.
        if database == 'records_db':
            database_to_use = self.records_db
        elif database == 'cache_records_db':
            database_to_use = self.cache_records_db
        else:
            raise ValueError("`database` must be one of 'records_db', 'cache_records_db'.")

        if type(database_to_use).__name__ != 'DataFrame':
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

        # Deploy and Return
        return db['raw_dicom_files_paths'].progress_map(dicom_apply, na_action='ignore')

    def pull(self,
             patient_limit=3,
             session_limit=1,
             collections_limit=None,
             allowed_modalities=None,
             image_format='png',
             save_dicoms=False,
             check_cache_first=True):
        """

        Pull (i.e., download) the current search.

        Notes:

        - 3D images are saved as individual frames.

        - Images have the following format:

            ``[instance, pull_position]__[patient_id_[Last 10 Digits of SeriesInstanceUID]]__[Image Scale ('default')].image_format``

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
        :param image_format: format for the image, e.g., 'png', 'jpg', etc. If ``None``, images will not be downloaded.
                           Defaults to 'png'.
        :type image_format: ``str``
        :param save_dicoms: if ``True``, save the raw dicom files. Defaults to ``False``.
        :type save_dicoms: ``bool``
        :param check_cache_first: check the image cache for the image prior to downloading.
                                  If the image is already present, no attempt will be made to download it again.

                 .. warning::

                        Manually deleting images from the cache is likely to interfere with this parameter.
                        For instance, if a single frame of a 3D image is missing from the cache the entire image will
                        be downloaded again.

        :type check_cache_first: ``bool``
        :return: a DataFrame with the record information.
        :rtype: ``Pandas DataFrame``
        """
        if self.current_query is None:
            raise TypeError("`current_query` is `None`: `search()` must be called before `pull()`.")

        # Note the time the pull request was made
        self._pull_time = datetime.now()

        # Download Records for all of the studies
        record_frames, pull_success = self._pull_records(patient_limit=patient_limit,
                                                         collections_limit=collections_limit)

        # Check for failures
        download_failures = [collection for (success, collection) in pull_success if success is False]

        if len(download_failures) == len(pull_success):
            raise IndexError("Data could not be harvested for any of the requested collections.")
        elif len(download_failures):
            warn("\n\nThe following collections failed to download:\n{0}".format(list_to_bulletpoints(download_failures)))

        # Combine all record frames
        records_db = pd.concat(record_frames, ignore_index=True)

        # Add the Version of BioVida which generated the DataFrame
        records_db['biovida_version'] = [__version__] * records_db.shape[0]

        # Download the images for all of the studies (collections)
        if isinstance(image_format, str):
            if self._verbose:
                print("\nObtaining Images...")
            self.records_db = self._Images.pull_images(records_db=records_db,
                                                       pull_time=self._pull_time.strftime(TIME_FORMAT),
                                                       session_limit=session_limit,
                                                       image_format=image_format,
                                                       save_dicoms=save_dicoms,
                                                       allowed_modalities=allowed_modalities,
                                                       check_cache_first=check_cache_first)
        else:
            self.records_db = records_db

        # Update the cache record if and only if a request was also made for images.
        if isinstance(image_format, str):
            self._tcia_cache_records_db_handler()

        return self.records_db




















