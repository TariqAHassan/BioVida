"""

    Cancer Imaging Archive Interface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
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
from itertools import chain
from datetime import datetime

from biovida.images._image_tools import dict_to_tot
from biovida.images._image_tools import load_temp_dbs
from biovida.images._image_tools import NoResultsFound
from biovida.images._image_tools import record_db_merge

from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import items_null
from biovida.support_tools.support_tools import only_numeric
from biovida.support_tools.support_tools import combine_dicts
from biovida.support_tools.support_tools import camel_to_snake_case
from biovida.support_tools.support_tools import list_to_bulletpoints

from biovida.support_tools.printing import pandas_pprint
from biovida.support_tools._cache_management import package_cache_creator
from biovida.images._resources.cancer_image_parameters import CancerImgArchiveParams


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
    # ToDo: add link to the TCIA page for a given collection/study (find a way using the default library).
    def __init__(self,
                 dicom_modality_abbrevs,
                 verbose=False,
                 cache_path=None,
                 tcia_homepage='http://www.cancerimagingarchive.net'):
        self._verbose = verbose
        self._tcia_homepage = tcia_homepage
        _, self._created_img_dirs = package_cache_creator(sub_dir='images', cache_path=cache_path, to_create=['aux'])
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
        save_path = os.path.join(self._created_img_dirs['aux'], 'all_tcia_studies.p')

        if not os.path.isfile(save_path) or download_override:
            if self._verbose:
                header("Downloading Record of Available Studies... ", flank=False)
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

        def modaility_filter(x, modaility):
            """Apply filter to look for rows which match `modaility`."""
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
            summary_df = summary_df[summary_df.apply(lambda x: modaility_filter(x, modality), axis=1)]

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
                                 "The separator being used to replace spaces in the URL is may incorrect. An attempt\n"
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
        :return: the name of 
        :rtype: ``str``
        """
        unique_studies = collection_series.unique()
        if len(unique_studies) == 1:
            collection = unique_studies[0]
        else:
            raise AttributeError("`{0}` studies found in `records`. Expected one.".format(str(len(unique_studies))))
        summary_df = self._Overview._all_studies_cache_mngt(download_override=overview_download_override)
        return summary_df[summary_df['collection'] == collection]['cancer_type'].iloc[0]

    def records_pull(self, study, search_dict, query_time, overview_download_override=False, patient_limit=3):
        """

        Extract record of all images for all patients in a given study.

        :param study: a Cancer Imaging Archive collection (study).
        :type study: ``str``
        :param search_dict: a dicitionary which contains the search information provided by the user
                            (as evolved inside  ``CancerImageInterface()_search_dict_gen()``.
        :type search_dict: ``dict``
        :param query_time: the time the query was launched.
        :type query_time: ``datetime``
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
        patient_study_df['query_time'] = [query_time] * patient_study_df.shape[0]

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

    def _real_time_update_db_path_gen(self, start_time, collection_name):
        """

        Define the path to be used when saving ``real_time_update_db``.

        :param start_time: see ``pull_img()``
        :type start_time: ``str``
        :param collection_name: the name of the collection (study).
        :type collection_name: ``str``
        """
        # Create the path name for ``real_time_update_db``.
        db_name = "{0}_{1}.p".format(start_time, cln(collection_name).replace(" ", "_"))
        self.real_time_update_db_path = os.path.join(self.temp_directory_path, db_name)

    def _save_real_time_update_db(self):
        """

        Save the ``real_time_update_db`` to disk.

        """
        # Save the `real_time_update_db` to disk.
        self.real_time_update_db.to_pickle(self.real_time_update_db_path)

    def _download_zip(self, series_uid, temporary_folder, index):
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
        # Define URL to extract the images from
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

    def _dicom_to_standard_image(self, f, pull_position, conversion, new_file_name, img_format):
        """

        This method handles the act of saving dicom images as in a more common file format (e.g., .png).
        An image (``f``) can be either 2 or 3 Dimensional.

        Notes:

        - 3D images will be saved as individual frames

        - if pydicom cannot render the DICOM as a pixel array, this method will its hault image extraction efforts.

        :param f: a (py)dicom image.
        :type f: ``pydicom object``
        :param pull_position: the position of the file in the list of files pulled from the database.
        :type pull_position: ``int``
        :param conversion: the color scale conversion to use, e.g., 'LA' or 'RGB'.
        :type conversion: ``str``
        :param new_file_name: see ``_save_dicom_as_img``'s ``save_name`` parameter.
        :type new_file_name: ``str``
        :param img_format: see: ``pull_img()``.
        :type img_format: ``str``
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
            file_name = "{0}__{1}__default.{2}".format(head, os.path.basename(new_file_name), img_format.replace(".", ""))
            return os.path.join(save_location, file_name)

        if pixel_arr.ndim == 2:
            # Define save name by combining the images instance in the set, `new_file_name` and `img_format`.
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
        current = self.real_time_update_db.get_value(index, column)
        old_iterable = current if isinstance(current, (list, tuple)) else []
        replacement = tuple(list(old_iterable) + list(new))
        self.real_time_update_db.set_value(index, column, tuple(list(old_iterable) + list(new)))
        # Return the length if requested.
        return len(replacement) if return_replacement_len else None

    def _save_dicom_as_img(self,
                           path_to_dicom_file,
                           index,
                           pull_position,
                           save_name=None,
                           color=False,
                           img_format='png'):
        """

        Save a dicom image as a more common file format.

        :param path_to_dicom_file: path to a dicom image
        :type path_to_dicom_file: ``str``
        :param pull_position: the position of the image in the raw zip data provided by the Cancer Imaging Archive API.
        :type pull_position: ``int``
        :param index: the row index currently being processed inside of the main loop in ``_pull_images_engine()``.
        :type index: ``int``
        :param save_name: name of the new file (do *NOT* include a file extension).
                          To specifiy a file format, use ``img_format``.
                          If ``None``, name from ``path_to_dicom_file`` will be conserved.
        :type save_name: ``str``
        :param color: If ``True``, convert the image to RGB before saving. If ``False``, save as a grayscale image.
                      Defaults to ``False``
        :type color: ``bool``
        :param img_format: see: ``pull_img()``.
        :type img_format: ``str``
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
        all_save_paths, success = self._dicom_to_standard_image(f, pull_position, conversion, new_file_name, img_format)

        # Update Record
        cfp_len = self._update_and_set_list(index, 'converted_files_paths', all_save_paths, return_replacement_len=True)
        self.real_time_update_db.set_value(index, 'image_count_converted_cache', cfp_len)

        # Add record of whether or not the dicom file could be converted to a standard image type
        self.real_time_update_db.set_value(index, 'conversion_success', success)

        # Save the data frame
        self._save_real_time_update_db()

    def _move_dicoms(self, save_dicoms, dicom_files, series_abbrev, index):
        """

        Move the dicom source files to ``self._created_img_dirs['dicoms']``.
        Employ to prevent the raw dicom files from being destroyed.

        :param save_dicoms: see: ``pull_img()``
        :type save_dicoms: ``bool``
        :param dicom_files: the yeild of ``_download_zip()``
        :type dicom_files: ``list``
        :param series_abbrev: as evolved inside ``_pull_images_engine()``.
        :type series_abbrev: ``str``
        :param index: the row index currently being processed inside of the main loop in ``_pull_images_engine()``.
        :type index: ``int``
        """
        if save_dicoms is False:
            self.real_time_update_db.set_value(index, 'raw_dicom_files_paths', np.NaN)
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

    def _valid_modaility(self, allowed_modalities, modality, modality_full):
        """

        Check if `modality` or `modality_full` contains the modality the user is looking for.

        :param allowed_modalities: see: ``pull_img()``
        :type allowed_modalities: ``list``, ``tuple`` or ``None``.
        :param modality: a single element from the ``modality`` column in ``self.real_time_update_db``.
        :type modality: ``str``
        :param modality_full: a single element from the ``modality_full`` column in ``self.real_time_update_db``.
        :type modality_full: ``str``
        :return: whether or not the image satisfies the modaility the user is looking for.
        :rtype: ``bool``
        """
        # Assume True if `allowed_modalities` is left to its default (`None`).
        if not isinstance(allowed_modalities, (list, tuple)):
            return True
        # Check if any item in `allowed_modalities` is a sublist in `modality` or `modality_full`.
        elif any([cln(l).lower() in cln(i).lower() for l in allowed_modalities for i in (modality, modality_full)]):
            return True
        else:
            return False

    def _pull_images_engine(self, save_dicoms, allowed_modalities, img_format, check_cache_first):
        """

        Tool to coordinate the above machinery for pulling and downloading images (or locating them in the cache).

        :param save_dicoms: see: ``pull_img()``.
        :type save_dicoms: ``bool``
        :param img_format: see: ``pull_img()``
        :param img_format: ``str``
        :param allowed_modalities: see: ``pull_img()``
        :type allowed_modalities: ``list``, ``tuple`` or ``None``.
        :param check_cache_first: see: ``pull_img()``
        :param check_cache_first: ``bool``
        """
        columns = ('series_instance_uid', 'patient_id', 'image_count', 'modality', 'modality_full')
        zipped_cols = list(zip(*[self.real_time_update_db[c] for c in columns] + [pd.Series(self.real_time_update_db.index)]))

        for series_uid, patient_id, image_count, modality, modality_full, index in tqdm(zipped_cols):
            # Check if the image should be harvested (or loaded from the cache).
            valid_image = self._valid_modaility(allowed_modalities, modality, modality_full)

            # Add whether or not the image was of the modaility (or modailities) requested by the user.
            self.real_time_update_db.set_value(index, 'allowed_modaility', valid_image)

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
                dicom_files = self._download_zip(series_uid, temporary_folder=temporary_folder, index=index)

                # Convert dicom files to `img_format`
                for e, f in enumerate(dicom_files, start=1):
                    self._save_dicom_as_img(f, index, pull_position=e, save_name=series_abbrev, img_format=img_format)

                # Save raw dicom files, if `save_dicoms` is True.
                self._move_dicoms(save_dicoms, dicom_files, series_abbrev, index)

                # Delete the temporary folder.
                shutil.rmtree(temporary_folder, ignore_errors=True)
            else:
                self._update_and_set_list(index, 'raw_dicom_files_paths', dsl_summary)
                self._update_and_set_list(index, 'converted_files_paths', sl_summary)
                self.real_time_update_db.set_value(index, 'conversion_success', cache_complete)
                self.real_time_update_db.set_value(index, 'image_count_converted_cache', len(sl_summary))
                # Save the data frame
                self._save_real_time_update_db()

    def pull_img(self,
                 collection_name,
                 records,
                 start_time,
                 session_limit=1,
                 img_format='png',
                 save_dicoms=True,
                 allowed_modalities=None,
                 check_cache_first=True):
        """

        Pull Images from the Cancer Imaging Archive.

        :param collection_name: name of the collection which ``records`` corresponds to.
        :type collection_name: ``str``
        :param records: the yeild from ``_CancerImgArchiveRecords().records_pull()``.
        :type records: ``Pandas DataFrame``
        :param start_time: the time the pull for images was initiated (standard format: "%Y_%h_%d__%H_%M_%S_%f").
        :type start_time: ``str``
        :param session_limit: restrict image harvesting to the first ``n`` sessions, where ``n`` is the value passed
                              to this parameter. If ``None``, no limit will be imposed. Defaults to `1`.
        :type session_limit: ``int``
        :param img_format: format for the image, e.g., 'png', 'jpg', etc. Defaults to 'png'.
        :type img_format: ``str``
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

        # Create the temp folder if it does not already exist.
        self._create_temp_directory()

        # Define the path to be used when saving ``real_time_update_db``.
        self._real_time_update_db_path_gen(start_time, collection_name)

        # Apply limit on number of sessions, if any
        if isinstance(session_limit, int):
            if session_limit < 1:
                raise ValueError("`session_limit` must be an intiger greater than or equal to 1.")
            img_records = records[records['session'].map(
                lambda x: float(x) <= session_limit if pd.notnull(x) else False)].reset_index(drop=True).copy(deep=True)

        # Add limited record
        self.real_time_update_db = img_records

        # Add columns which will be populated as the images are pulled.
        for c in ('raw_dicom_files_paths', 'converted_files_paths', 'conversion_success',
                  'allowed_modaility', 'image_count_converted_cache'):
            self.real_time_update_db[c] = None

        # Harvest images
        self._pull_images_engine(save_dicoms, allowed_modalities, img_format, check_cache_first)
        self.real_time_update_db = self.real_time_update_db.replace({None: np.NaN})

        return self.real_time_update_db


# ----------------------------------------------------------------------------------------------------------
# Construct Database
# ----------------------------------------------------------------------------------------------------------


class CancerImageInterface(object):
    """

    Python Interface for the Cancer Imaging Archive's API.

    :param api_key: an key to the the Cancer Imaging Archive's API.
                    To request a key, please see:
            https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+%28REST+API%29+Usage+Guide
    :type api_key: ``str``
    :param verbose: print additional details.
    :type verbose: ``bool``
    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                       one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    """

    def _tcia_record_db_gen(self, tcia_record_db_addition):
        """

        Generate the `tcia_record_db` database.
        If it does not exist, use `tcia_record_db`.
        If if already exists, merge it with `tcia_record_db_addition`.

        :param tcia_record_db_addition: the new search dataframe to added to the existing one.
        :type tcia_record_db_addition: ``Pandas DataFrame``
        """
        # Compose or update the master 'tcia_record_db' dataframe
        if self.tcia_record_db is None:
            self.tcia_record_db = tcia_record_db_addition.copy(deep=True)
            self.tcia_record_db.to_pickle(self._tcia_record_db_save_path)
        else:
            duplicates_subset_columns = [c for c in self.tcia_record_db.columns if c != 'query_time']
            self.tcia_record_db = record_db_merge(current_record_db=self.tcia_record_db,
                                                  record_db_addition=tcia_record_db_addition,
                                                  query_column_name='query',
                                                  query_time_column_name='query_time',
                                                  duplicates_subset_columns=duplicates_subset_columns,
                                                  sort_on=['query_time', 'study_name'],
                                                  relationship_mapping_func=None)

            # Save to disk
            self.tcia_record_db.to_pickle(self._tcia_record_db_save_path)

    def __init__(self,
                 api_key,
                 verbose=True,
                 cache_path=None):
        self._verbose = verbose
        self.dicom_modality_abbrevs = CancerImgArchiveParams().dicom_modality_abbreviations('dict')

        # Root URL to for the Cancer Imaging Archive's REST API
        root_url = 'https://services.cancerimagingarchive.net/services/v3/TCIA'

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

        # DataFrames
        self.current_search = None
        self.current_db = None

        # The format for date information
        self._time_format = "%Y_%h_%d__%H_%M_%S_%f"

        # Path to the tcia_record_db
        self._tcia_record_db_save_path = os.path.join(self._Images._created_img_dirs['databases'], 'tcia_record_db.p')

        # Load `tcia_record_db` if it exists already, else set to None.
        if os.path.isfile(self._tcia_record_db_save_path):
            self.tcia_record_db = pd.read_pickle(self._tcia_record_db_save_path)
            # Merge any latent elements in the __temp__ folder, if such a folder exists (and if it is populated).
            if os.path.isdir(self._Images.temp_directory_path):
                latent_temps = load_temp_dbs(self._Images.temp_directory_path)
                if latent_temps is not None:
                    self._tcia_record_db_gen(tcia_record_db_addition=latent_temps)
                # Delete the latent '__temp__' folder
                shutil.rmtree(self._Images.temp_directory_path, ignore_errors=True)
        else:
            self.tcia_record_db = None

        # Dictionary of the most recent search
        self._search_dict = None
        self._query_time = None

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

        self._search_dict = {k: lower_sdict(v) for k, v in sdict.items()}

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
        :param pretty_print: if ``True``, pretty print the search results. Defaults to ``True``.
        :type pretty_print: ``bool``
        :return: a dataframe containing the search results.
        :rtype: ``Pandas DataFrame``

        :Example:

        >>> CancerImageInterface().search(cancer_type='carcinoma', location=['head', 'neck'])
        ...
           collection               cancer_type               Modalities   Subjects    Location    Metadata  ...
        0  TCGA-HNSC  Head and Neck Squamous Cell Carcinoma  CT, MR, PT     164     [Head, Neck]    Yes     ...

        """
        # Note the time the search request was made
        self._query_time = datetime.now()

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
        self.current_search = summary_df.reset_index(drop=True)

        if pretty_print:
            current_search_print = self.current_search.copy(deep=True)
            pandas_pprint(current_search_print, full_cols=True, col_align='left', lift_column_width_limit=True)

        # Warn the user if search criteria have not been applied.
        if all([collection is None, cancer_type is None, location is None, modality is None]):
            sleep(0.25)
            warn("\nSpecific search criteria have not been applied.\n"
                 "If `pull()` is called, *all* collections will be downloaded.\n"
                 "Such a request could yield several terabytes of data.\n"
                 "If you still wish to proceed, consider adjusting `pull()`'s\n"
                 "`patient_limit` and `session_limit` parameters.")

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
            all_collections = self.current_search['collection'][:collections_limit]
        else:
            all_collections = self.current_search['collection']

        # Loop through and download all of the studies
        record_frames = list()
        for collection in all_collections:
            if self._verbose:
                print("\nDownloading records for the '{0}' collection...".format(collection))
            try:
                record_frames.append(self._Records.records_pull(study=collection,
                                                                search_dict=self._search_dict,
                                                                query_time=self._query_time,
                                                                patient_limit=patient_limit))
                pull_success.append((True, collection))
            except IndexError as e:
                warn("\nIndexError Encountered: {0}".format(e))
                pull_success.append((False, collection))

        return record_frames, pull_success

    def _pull_images(self,
                     record_frames,
                     collection_names,
                     session_limit,
                     img_format,
                     save_dicoms,
                     allowed_modalities,
                     check_cache_first):
        """

        Pull Images from the TCIA API (based on the 'records' obtained with ``_pull_records()``.

        :param record_frames: a list of records dataframes.
        :type record_frames: ``list``
        :param collection_names: list of strings noting collects for which records were sucessfully downloaded.
        :type collection_names: ``list``
        :param session_limit: restrict image harvesting to the first ``n`` sessions, where ``n`` is the value passed
                              to this parameter. If ``None``, no limit will be imposed. Defaults to 1.
        :type session_limit: ``int``
        :param img_format: see: ``pull()``
        :type img_format: ``str``
        :param save_dicoms: see: ``pull()``.
        :type save_dicoms: ``bool``
        :param allowed_modalities: see: ``pull()``.
        :type allowed_modalities: ``list`` or ``tuple``
        :param check_cache_first: see: ``pull()``.
        :type check_cache_first: ``bool``
        :return: a list of dataframes passed through ``_CancerImgArchiveImages().pull_img()``.
        :rtype: ``list``
        """
        img_frames = list()
        for records, collection in zip(record_frames, collection_names):
            if self._verbose:
                print("\nObtaining images for the '{0}' collection...".format(collection))
            current_img_frame = self._Images.pull_img(collection_name=collection,
                                                      records=records,
                                                      start_time=self._query_time.strftime(self._time_format),
                                                      session_limit=session_limit,
                                                      img_format=img_format,
                                                      save_dicoms=save_dicoms,
                                                      allowed_modalities=allowed_modalities,
                                                      check_cache_first=check_cache_first)
            img_frames.append(current_img_frame)

        return img_frames

    def _tcia_record_db_handler(self):
        """

        Behavior:

        If ``tcia_record_db`` does not exists on disk, create it using ``tcia_record_db_addition`` (evolved inside this
        function). If it already exists, merge it with ``tcia_record_db_addition``.

        More broadly, this whole script follows the following procedure for caching:

        1. Create a master dataframe. If none exists, created one based on the steps below.
        2. Create a temporary folder which will be populated with temporary databases which update in real time as
           images are pulled.
        3. Upon sucessful exit in ``CancerImageInterface().pull()``, combine all temp. dataframes and merge them with
           the master and run the pruning algorithm (see the second half of ``_tcia_record_db_gen()``).
        4. As a precaution, on init. of the CancerImageInterface class, if a '__temp__' folder exists,
           merge and run the pruning algorithm.

        """
        # Generate the `tcia_record_db`.
        self._tcia_record_db_gen(tcia_record_db_addition=load_temp_dbs(self._Images.temp_directory_path))

        # Delete the '__temp__' folder
        shutil.rmtree(self._Images.temp_directory_path, ignore_errors=True)

    def pull(self,
             patient_limit=3,
             session_limit=1,
             collections_limit=None,
             allowed_modalities=None,
             img_format='png',
             save_dicoms=False,
             check_cache_first=True):
        """

        Pull (i.e., download) the current search.

        Notes:

        - 3D images are saved as individual frames.

        - Images have the following format:

            ``[instance, pull_position]__[patient_id_[Last 10 Digits of SeriesUID]]__[Image Scale (D=Default)].img_format``

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

                        Several studies in the Cancer Imaging Archive database contain studies with multiple imaging
                        sessions. Latter sessions may be of patients following various forms of intervention,
                        such as surgery, intended to *eliminate* cancerous tissue. For this reason, it cannot be assumed
                        that images  obtained from non-baseline sessions (i.e., session number > 1) do, in fact, contain
                        signs of disease.

        :type session_limit: ``int``
        :param collections_limit: limit the number of collections to download. If ``None``, no limit will be applied.
                                  Defaults to ``None``.
        :type collections_limit: ``int`` or ``None``
        :param allowed_modalities: limit images downloaded to certain modalities.
                                   See: CancerImageInterface().dicom_modality_abbrevs (use the keys).
                                   Note: 'MRI', 'PET', 'CT' and 'X-Ray' can also be used.
                                   This parameter is not case sensitive. Defaults to ``None``.
        :type allowed_modalities: ``list`` or ``tuple``
        :param img_format: format for the image, e.g., 'png', 'jpg', etc. If ``None``, images will not be downloaded.
                           Defaults to 'png'.
        :type img_format: ``str``
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
        if self.current_search is None:
            raise AttributeError("`current_search` is empty. A search must be performed before `pull()` is called.")

        # Download Records for all of the studies
        record_frames, pull_success = self._pull_records(patient_limit=patient_limit,
                                                         collections_limit=collections_limit)

        # Check for failures
        download_failures = [collection for (success, collection) in pull_success if success is False]
        download_successes = [collection for (success, collection) in pull_success if success is True]

        if len(download_failures) == len(pull_success):
            raise IndexError("Data could not be harvested for any of the requested collections.")
        elif len(download_failures):
            warn("\nThe following collections failed to download:\n{0}".format(list_to_bulletpoints(download_failures)))

        # Download the images for all of the studies
        if isinstance(img_format, str):
            final_frames = self._pull_images(record_frames=record_frames,
                                             collection_names=download_successes,
                                             session_limit=session_limit,
                                             img_format=img_format,
                                             save_dicoms=save_dicoms,
                                             allowed_modalities=allowed_modalities,
                                             check_cache_first=check_cache_first)
        else:
            final_frames = record_frames

        # Concatenate
        self.current_db = pd.concat(final_frames, ignore_index=True)

        # Update the image record if and only if a request was also made for images.
        if isinstance(img_format, str):
            self._tcia_record_db_handler()

        return self.current_db




























