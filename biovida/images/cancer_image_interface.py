"""

    The Cancer Imaging Archive Interface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import io
import os
import dicom
import pickle
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from warnings import warn
from itertools import chain
from time import sleep

from biovida.images.ci_api_key import API_KEY

from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import only_numeric
from biovida.support_tools.support_tools import combine_dicts
from biovida.support_tools.support_tools import items_null
from biovida.support_tools.printing import pandas_pprint

from biovida.support_tools._cache_management import package_cache_creator
from biovida.images._resources.cancer_image_parameters import CancerImgArchiveParams


# ----------------------------------------------------------------------------------------------------------
# Summarize Studies Provided Through The Cancer Imaging Archive
# ----------------------------------------------------------------------------------------------------------


class _CancerImgArchiveOverview(object):
    """

    Overview of Information Available on The Cancer Imaging Archive.

    :param cache_path: path to the location of the BioVida cache. If a cache does not exist in this location,
                        one will created. Default to ``None``, which will generate a cache in the home folder.
    :type cache_path: ``str`` or ``None``
    :param verbose: if ``True`` print additional information. Defaults to ``False``.
    :type verbose: ``bool``
    :param tcia_homepage: URL to the The Cancer Imaging Archive's homepage.
    :type tcia_homepage: ``str``
    """

    def __init__(self, dicom_modaility_abbrevs, verbose=False, cache_path=None, tcia_homepage='http://www.cancerimagingarchive.net'):
        self._verbose = verbose
        self._tcia_homepage = tcia_homepage
        _, self._created_img_dirs = package_cache_creator(sub_dir='images', cache_path=cache_path, to_create=['aux'])
        self.dicom_modaility_abbrevs = dicom_modaility_abbrevs

    def _all_studies_parser(self):
        """

        Get a record of all studies on The Cancer Imaging Archive.

        :return: the table on the homepage
        :rtype: ``Pandas DataFrame``
        """
        # Extract the main summary table from the home page
        summary_df = pd.read_html(str(requests.get(self._tcia_homepage).text), header=0)[0]

        # Drop Studies which are 'Coming Soon'.
        summary_df = summary_df[summary_df['Status'].str.strip().str.lower() != 'coming soon']

        # Drop Studies which are on phantoms
        summary_df = summary_df[~summary_df['Location'].str.lower().str.contains('phantom')]

        # Drop Studies which are on mice or phantoms
        summary_df = summary_df[~summary_df['Collection'].str.lower().str.contains('mouse|phantom')]

        # Only Keep Studies which are public
        summary_df = summary_df[summary_df['Access'].str.strip().str.lower() == 'public'].reset_index(drop=True)

        # Add Full Name for Modalities
        summary_df['ModalitiesFull'] = summary_df['Modalities'].map(
            lambda x: [self.dicom_modaility_abbrevs.get(cln(i), i) for i in cln(x).split(", ")])

        # Parse the Location Column (and account for special case: 'Head-Neck').
        summary_df['Location'] = summary_df['Location'].map(
            lambda x: cln(x.replace(" and ", ", ").replace("Head-Neck", "Head, Neck")).split(", "))

        # Convert 'Update' to Datetime
        summary_df['Updated'] = pd.to_datetime(summary_df['Updated'], infer_datetime_format=True)

        # Clean Column names
        summary_df.columns = list(map(lambda x: cln(x, extent=2), summary_df.columns))

        return summary_df

    def _all_studies_cache_mngt(self, download_override):
        """

        Obtain and Manage a copy the table which summarizes the The Cancer Imaging Archive
        on the organization's homepage.

        :param download_override: If ``True``, override any existing database currently cached and download a new one.
        :type download_override: ``bool``
        :return: summary table hosted on the home page of The Cancer Imaging Archive.
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
            summary_df = summary_df[summary_df['CancerType'].str.lower().str.contains(cancer_type)]

        # Filter by `location`
        if isinstance(location, (str, list, tuple)):
            location = [location] if isinstance(location, str) else location
            summary_df = summary_df[summary_df['Location'].map(
                lambda x: any([cln(l).lower() in i.lower() for i in x for l in location]))]

        # Filter by `modality`.
        if isinstance(modality, (str, list, tuple)):
            modality = [modality] if isinstance(modality, str) else modality
            summary_df = summary_df[summary_df['Modalities'].map(
                lambda x: any([cln(m).lower() in i.lower() for i in cln(x).split(", ") for m in modality]))]

        return summary_df


# ----------------------------------------------------------------------------------------------------------
# Pull Records from The Cancer Imaging Archive
# ----------------------------------------------------------------------------------------------------------


class _CancerImgArchiveRecords(object):
    """

    :param root_url: the root URL for the The Cancer Imaging Archive's API.
    :type root_url: ``str``
    """

    def __init__(self, dicom_modaility_abbrevs, cancer_img_archive_overview, root_url):
        self._root_url = root_url
        self.records_df = None
        self.dicom_modaility_abbrevs = dicom_modaility_abbrevs
        self._Overview = cancer_img_archive_overview

    def _extract_study(self, study):
        """

        Download all patients in a given study.

        :param study:
        :type study: ``str``
        :return:
        """
        url = '{0}/query/getPatientStudy?Collection={1}&format=csv&api_key={2}'.format(
            self._root_url, cln(study).replace(' ', '+'), API_KEY)
        return pd.DataFrame.from_csv(url).reset_index()

    def _date_index_map(self, list_of_dates):
        """

        Returns a dict of the form: ``{date: index in ``list_of_dates``, ...}``

        :param list_of_dates:
        :type list_of_dates:
        :return:
        """
        return {k: i for i, k in enumerate(sorted(list_of_dates), start=1)}

    def _summarize_study_by_patient(self, study):
        """

        Summarizes a study by patient.
        Note: patient_limits summary to baseline (i.e., follow ups are excluded).

        :param study:
        :type study: ``str``
        :return: nested dictionary of the form:

                ``{PatientID: {StudyInstanceUID: {'sex':..., 'age': ..., 'session': ..., 'StudyDate': ...}}}``

        :rtype: ``dict``
        """
        # Download a summary of all patients in a study
        study_df = self._extract_study(study)

        # Convert StudyDate to datetime
        study_df['StudyDate'] = pd.to_datetime(study_df['StudyDate'], infer_datetime_format=True)

        # Divide Study into stages (e.g., Baseline (session 1); Baseline + 1 Month (session 2), etc.
        stages = study_df.groupby('PatientID').apply(lambda x: self._date_index_map(x['StudyDate'].tolist())).to_dict()

        # Apply stages
        study_df['Session'] = study_df.apply(lambda x: stages[x['PatientID']][x['StudyDate']], axis=1)

        # Define Columns to Extract from study_df
        valuable_cols = ('PatientID', 'StudyInstanceUID', 'Session', 'PatientSex', 'PatientAge', 'StudyDate')

        # Convert to a nested dictionary
        patient_dict = dict()
        for pid, si_uid, session, sex, age, date in zip(*[study_df[c] for c in valuable_cols]):
            inner_nest = {'sex': sex, 'age': age, 'session': session, 'StudyDate': date}
            if pid not in patient_dict:
                patient_dict[pid] = {si_uid: inner_nest}
            else:
                patient_dict[pid] = combine_dicts(patient_dict[pid], {si_uid: inner_nest})

        return patient_dict

    def _patient_img_summary(self, patient, study, patient_dict):
        """

        Harvests the Cancer Image Archive's Text Record of all baseline images for a given patient
        in a given study.

        :param patient:
        :return:
        """
        # Select an individual Patient
        url = '{0}/query/getSeries?Collection={1}&PatientID={2}&format=csv&api_key={3}'.format(
            self._root_url, cln(study).replace(' ', '+'), patient, API_KEY)
        patient_df = pd.DataFrame.from_csv(url).reset_index()

        def upper_first(s):
            return "{0}{1}".format(s[0].upper(), s[1:])

        # Add Sex, Age, Session, and StudyDate
        patient_info = patient_df['StudyInstanceUID'].map(
            lambda x: {upper_first(k): patient_dict[x][k] for k in ('sex', 'age', 'session', 'StudyDate')})
        patient_df = patient_df.join(pd.DataFrame(patient_info.tolist()))

        # Add PatientID
        patient_df['PatientID'] = patient

        return patient_df

    def _clean_patient_study_df(self, patient_study_df):
        """

        Cleans the input in the following ways:

            - convert 'F' --> 'Female' and 'M' --> 'Male'

            - Converts the 'Age' column to numeric (years)

            - Remove line breaks in the 'ProtocolName' and 'SeriesDescription' columns

            - Add Full name for modality (ModalityFull)

            - Convert the 'SeriesDate' column to datetime

        :param patient_study_df: the ``patient_study_df`` dataframe evolved inside ``_pull_records()``.
        :type patient_study_df: ``Pandas DataFrame``
        :return: a cleaned ``patient_study_df``
        :rtype: ``Pandas DataFrame``
        """
        # convert 'F' --> 'female' and 'M' --> 'male'.
        patient_study_df['Sex'] = patient_study_df['Sex'].map(
            lambda x: {'F': 'female', 'M': 'male'}.get(cln(str(x)).upper(), x), na_action='ignore')

        # Convert entries in the 'Age' Column to floats.
        patient_study_df['Age'] = patient_study_df['Age'].map(
            lambda x: only_numeric(x) / 12.0 if 'M' in str(x).upper() else only_numeric(x), na_action='ignore')

        # Remove unneeded line break marker
        for c in ('ProtocolName', 'SeriesDescription'):
            patient_study_df[c] = patient_study_df[c].map(lambda x: cln(x.replace("\/", " ")), na_action='ignore')

        # Add the full name for modality.
        patient_study_df['ModalityFull'] = patient_study_df['Modality'].map(
            lambda x: self.dicom_modaility_abbrevs.get(x, np.NaN), na_action='ignore')

        # Convert SeriesDate to datetime
        patient_study_df['SeriesDate'] = pd.to_datetime(patient_study_df['SeriesDate'], infer_datetime_format=True)

        # Sort and Return
        return patient_study_df.sort_values(by=['PatientID', 'Session']).reset_index(drop=True)

    def _get_illness_name(self, collection_series, overview_download_override):
        """

        :param collection:
        :return:
        """
        unique_studies = collection_series.unique()
        if len(unique_studies) == 1:
            collection = unique_studies[0]
        else:
            raise AttributeError("`{0}` studies found in `records`.".format(str(len(unique_studies))))
        summary_df = self._Overview._all_studies_cache_mngt(download_override=overview_download_override)
        return summary_df[summary_df['Collection'] == collection]['CancerType'].iloc[0]

    def records_pull(self, study, overview_download_override=False, patient_limit=3):
        """

        Extract record of all images for all patients in a given study.

        :param study:
        :type study: ``str``
        :param patient_limit: patient_limit on the number of patients to extract.
                             Patient IDs are sorted prior to this patient_limit being imposed.
                             If ``None``, no patient_limit will be imposed. Defaults to `3`.
        :type patient_limit: ``int`` or ``None``
        :return: a dataframe of all baseline images
        :rtype: ``Pandas DataFrame``
        """
        # ToDo: add illness name to dataframe.
        # ToDo: consider adding record dataframe cacheing
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
        patient_study_df['StudyName'] = study

        # Add the Name of the illness
        patient_study_df['CancerType'] = self._get_illness_name(patient_study_df['Collection'], overview_download_override)

        # Clean the dataframe
        self.records_df = self._clean_patient_study_df(patient_study_df)

        return self.records_df


# ----------------------------------------------------------------------------------------------------------
# Pull Images from The Cancer Imaging Archive
# ----------------------------------------------------------------------------------------------------------


class _CancerImgArchiveImages(object):
    """

    :param root_url: the root URL for the The Cancer Imaging Archive's API.
    :type root_url: ``str``
    """

    def __init__(self, dicom_modaility_abbrevs, root_url, cache_path=None):
        _, self._created_img_dirs = package_cache_creator(sub_dir='images', cache_path=cache_path,
                                                          to_create=['dicoms', 'raw'])
        self._root_url = root_url
        self.dicom_modaility_abbrevs = dicom_modaility_abbrevs

    def _download_zip(self, series_uid, temporary_folder):
        """

        :param series_uid:
        :param temporary_folder:
        :return: list of paths to the new files.
        """
        # See: http://stackoverflow.com/a/14260592/4898004

        # Define URL to extract the images from
        url = '{0}/query/getImage?SeriesInstanceUID={1}&format=csv&api_key={2}'.format(
            self._root_url, series_uid, API_KEY)
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(temporary_folder)
        def file_path_full(f):
            base_name = cln(os.path.basename(f.filename))
            return os.path.join(temporary_folder, base_name) if len(base_name) else None
        return list(filter(None, [file_path_full(f) for f in z.filelist]))

    def _dicom_to_standard_image(self, f, pull_position, conversion, new_file_name, img_format):
        """

        This method handles the act of saving dicom images as in a more common file format (e.g., .png).
        An image (``f``) can be either 2 or 3 Dimensional.

        :param f: a dicom image.
        :type f:
        :param pull_position: the position of the file in the list of files pulled from the database.
        :type pull_position: ``int``
        :param conversion:
        :param new_file_name:
        :type new_file_name: ``str``
        :param img_format:
        :type img_format: ``str``
        :return: tuple of the form: ``(a list of paths to saved images, boolean denoting success)``
        :rtype: ``tuple``
        """
        # Define a list to populate with a record of all images saved
        all_save_paths = list()

        # Extract a pixel array from the dicom file.
        try:
            pixel_arr = f.pixel_array
        except UnboundLocalError:  # ToDO: change to TypeError pending https://github.com/darcymason/pydicom/pull/309
            return [], False

        save_location = self._created_img_dirs['raw']
        def save_path(instance):
            """Define the path to save the image to."""
            head = "{0}_{1}".format(instance, pull_position)
            file_name = "{0}__{1}__D.{2}".format(head, os.path.basename(new_file_name), img_format.replace(".", ""))
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
            raise ValueError("Cannot coerce {0} dimensional image arrays. Images must be 2D or 3D.".format(pixel_arr.ndim))

        return all_save_paths, True

    def _save_dicom_as_img(self, path_to_dicom_file, pull_position, save_name=None, color=False, img_format='png'):
        """

        Save a dicom image as a more common file format.

        :param path_to_dicom_file: path to a dicom image
        :type path_to_dicom_file: ``str``
        :param pull_position:
        :type pull_position: ``int``
        :param save_name: name of the new file (do *NOT* include a file extension).
                          To specifiy a file format, use ``img_format``.
                          If ``None``, name from ``path_to_dicom_file`` will be conserved.
        :type save_name: ``str``
        :param color: If ``True``, convert the image to RGB before saving. If ``False``, save as a grayscale image.
                      Defaults to ``False``
        :type color: ``bool``
        :param img_format: format for the image, e.g., 'png', 'jpg', etc. Defaults to 'png'.
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

        # Convert the image into a PIL object and Save
        return self._dicom_to_standard_image(f, pull_position, conversion, new_file_name, img_format)

    def _move_dicoms(self, dicom_files, series_abbrev):
        """

        Move the dicom source files to ``self._created_img_dirs['dicoms']``.
        Employ to prevent the raw dicom files from being destroyed.

        :param dicom_files:
        :param series_abbrev:
        :return:
        """
        new_dircom_paths = list()
        for f in dicom_files:
            # Define a name for the new file by extracting the dicom file name and combining with `series_abbrev`.
            f_parsed = list(os.path.splitext(os.path.basename(f)))
            new_dicom_file_name = "{0}__{1}{2}".format(f_parsed[0], series_abbrev, f_parsed[1])

            # Define the location of the new files
            new_location = os.path.join(self._created_img_dirs['dicoms'], new_dicom_file_name)
            new_dircom_paths.append(new_location)

            # Move the dicom file from __temp__ --> to --> new location
            os.rename(f, new_location)

        return tuple(new_dircom_paths)

    def _cache_check(self, check_cache_first, series_abbrev, n_images_min, save_dicoms):
        """

        Check that caches likely contain that data which would be obtained by downloading it from the database.

        :param series_abbrev:
        :return: tuple of the form:

                ``(cache likely complete,
                   series_abbrev matches in self._created_img_dirs['raw'],
                   series_abbrev matches in self._created_img_dirs['dicoms'])``

        :type: ``tuple``
        :param n_images_min:
        :type n_images_min: ``int``
        """
        # Instruct ``_pull_images_engine()`` to download the images without checking the cache first.
        if check_cache_first is False:
            return False, None, None

        # Check that `self._created_img_dirs['raw']` has files which contain the string `series_abbrev`.
        save_location_summary = [f for f in os.listdir(self._created_img_dirs['raw']) if series_abbrev in f]

        # Check that `self._created_img_dirs['dicoms'])` has files which contain the string `series_abbrev`.
        dicoms_save_location_summary_complete = False
        if save_dicoms:
            dicoms_save_location_summary = [f for f in os.listdir(self._created_img_dirs['dicoms']) if series_abbrev in f]
            dicoms_save_location_summary_complete = len(dicoms_save_location_summary) >= n_images_min
        else:
            dicoms_save_location_summary = np.NaN
            dicoms_save_location_summary_complete = True

        # Compose completeness boolean from the status of
        # `self._created_img_dirs['raw']` and `self._created_img_dirs['dicoms']`
        complete = len(save_location_summary) >= n_images_min and dicoms_save_location_summary_complete

        return complete, save_location_summary, dicoms_save_location_summary

    def _create_temp_dir(self, temp_folder_name='__temp__'):
        """

        Create a temporary directory.

        :param temp_folder_name:
        :return:
        """
        temp_folder = os.path.join(self._created_img_dirs['dicoms'], temp_folder_name)
        if os.path.isdir(temp_folder):
            shutil.rmtree(temp_folder, ignore_errors=True)
        os.makedirs(temp_folder)
        return temp_folder

    def _pull_images_engine(self, img_records, save_dicoms, img_format, check_cache_first):
        """

        :param img_records:
        :param save_dicoms:
        :param img_format:
        :param check_cache_first:
        :return:
        """
        # ToDo: consider extracting patient weight from the dicom file.
        # ToDo: add real-time record keeping for files as they are downloaded.
        converted_files, conversion_success, raw_dicom_files = list(), list(), list()

        # Note: tqdm appears to be unstable with generators (hence `list()`).
        pairings = list(zip(*[img_records[c] for c in ('SeriesInstanceUID', 'PatientID', 'ImageCount')]))
        for series_uid, patient_id, image_count in tqdm(pairings):
            # Compose central part of the file name from 'PatientID' and the last ten digits of 'SeriesInstanceUID'
            series_abbrev = "{0}_{1}".format(patient_id, str(series_uid)[-10:])

            # Analyze the cache to determine whether or not downloading the images is warented
            cache_complete, sl_summary, dsl_summary = self._cache_check(check_cache_first, series_abbrev,
                                                                        image_count, save_dicoms)

            if not cache_complete:
                # Create temp. foloder
                temporary_folder = self._create_temp_dir()

                # Download the images into a temp. folder
                dicom_files = self._download_zip(series_uid, temporary_folder=temporary_folder)

                # Convert dicom files to `img_format`
                cfs = [self._save_dicom_as_img(f, pull_position=e, save_name=series_abbrev, img_format=img_format)
                       for e, f in enumerate(dicom_files, start=1)]
                converted_files.append([i[0] for i in cfs])
                conversion_success += [i[1] for i in cfs]

                # Save raw dicom files
                raw_dicom_files.append(self._move_dicoms(dicom_files, series_abbrev) if save_dicoms else np.NaN)

                # Delete the temp folder.
                shutil.rmtree(temporary_folder, ignore_errors=True)
            else:
                converted_files.append([sl_summary])
                raw_dicom_files.append(tuple(dsl_summary))
                conversion_success.append(True)

        def cf_flatten():
            """Flatten the inner most dimension of `converted_files`."""
            to_return = list()
            for cf in converted_files:
                flat = tuple(chain(*cf))
                to_return.append(flat if len(flat) else np.NaN)
            return to_return

        # Return the position of all files
        return cf_flatten(), raw_dicom_files, conversion_success

    def pull_img(self,
                    records,
                    session_limit=1,
                    img_format='png',
                    save_dicoms=True,
                    check_cache_first=True):
        """

        :param records:
        :param session_limit: restruct image harvesting to the first ``n`` sessions, where ``n`` is the value passed
                              to this parameter. If ``None``, no limit will be imposed. Defaults to 1.
        :type session_limit: ``int``
        :param img_format:
        :param save_dicoms:
        :param check_cache_first:
        :return:
        """
        # Apply limit on number of sessions, if any
        if isinstance(session_limit, int):
            if session_limit < 1:
                raise ValueError("`session_limit` must be an intiger greater than or equal to 1.")
            img_records = records[records['Session'].map(
                lambda x: float(x) <= session_limit if pd.notnull(x) else False)].reset_index(drop=True)

        # Harvest images
        converted_files, raw_dicom_files, conversion_success = self._pull_images_engine(img_records, save_dicoms,
                                                                                        img_format, check_cache_first)

        # Add paths to the images
        img_records['ConvertedFilesPaths'] = converted_files
        img_records['RawDicomFilesPaths'] = raw_dicom_files

        # Add record of whether or not the dicom file could be converted to a standard image type
        img_records['ConversionSuccess'] = conversion_success

        # Add column which provides the number of images each SeriesInstanceUID yeilded
        # Note: this may be discrepant with the 'ImageCount' column because 3D images are expanded into
        #       their individual frames when saved to the converted images cache.
        img_records['ImageCountConvertedCache'] = img_records['ConvertedFilesPaths'].map(
            lambda x: len(x) if not items_null(x) else 0)

        return img_records


# ----------------------------------------------------------------------------------------------------------
# Construct Database
# ----------------------------------------------------------------------------------------------------------


class CancerImageInterface(object):
    """

    :param verbose:
    :param cache_path:
    :param root_url:
    """

    def __init__(self,
                 verbose=True,
                 cache_path=None,
                 root_url='https://services.cancerimagingarchive.net/services/v3/TCIA'):
        self._verbose = verbose
        self.dicom_modaility_abbrevs = CancerImgArchiveParams().dicom_modality_abbreviations('dict')

        self._Overview = _CancerImgArchiveOverview(self.dicom_modaility_abbrevs, verbose, cache_path=cache_path)
        self._Records = _CancerImgArchiveRecords(self.dicom_modaility_abbrevs, self._Overview, root_url)
        self._Images = _CancerImgArchiveImages(self.dicom_modaility_abbrevs, cache_path=cache_path, root_url=root_url)

        # DataFrames
        self.current_search = None
        self.pull_records = None

    def _collection_filter(self, summary_df, collection, cancer_type, location):
        """

        Limits `summary_df` to an individual collection.

        :param summary_df:
        :param collection:
        :param cancer_type:
        :param location:
        :return:
        """
        # Filter by `collection`
        if isinstance(collection, str) and any(i is not None for i in (cancer_type, location)):
            raise ValueError("Both `cancer_types` and `location` must be ``None`` if a `collection` name is passed.")
        elif isinstance(collection, str):
            summary_df = summary_df[summary_df['Collection'].str.strip().str.lower() == collection.strip().lower()]
            if summary_df.shape[0] == 0:
                raise AttributeError("No Collection with the name '{0}' could be found.".format(collection))
            else:
                return summary_df.reset_index(drop=True)

    def search(self,
               collection=None,
               cancer_type=None,
               location=None,
               modality=None,
               download_override=False,
               pretty_print=True):
        """

        Method to Search for studies on The Cancer Imaging Archive.

        :param collection: a collection (study) hosted by The Cancer Imaging Archive.
        :type collection: ``str`` or ``None``
        :param cancer_type: a string or list/tuple of specifying cancer types.
        :type cancer_type: ``str``, ``iterable`` or ``None``
        :param location: a string or list/tuple of specifying body locations.
        :type location: ``str``, ``iterable`` or ``None``
        :param modality:
        :type modality: ``str``, ``iterable`` or ``None``
        :param download_override: If ``True``, override any existing database currently cached and download a new one.
                                  Defaults to ``False``.
        :param pretty_print: if ``True``, pretty print the search results. Defaults to ``True``.
        :type pretty_print: ``bool``
        :return: a dataframe containing the search results.
        :rtype: ``Pandas DataFrame``

        :Example:

        >>> _CancerImgArchiveOverview().studies(cancer_type=['Squamous'], location=['head'])
        ...
           Collection               CancerType               Modalities   Subjects    Location    Metadata  ...
        0  TCGA-HNSC  Head and Neck Squamous Cell Carcinoma  CT, MR, PT     164     [Head, Neck]    Yes     ...

        """
        # Load the Summary Table
        summary_df = self._Overview._all_studies_cache_mngt(download_override)

        # Filter by `collection`
        if collection is not None:
            summary_df = self._collection_filter(summary_df, collection, cancer_type,location)
        else:
            # Apply Filters
            summary_df = self._Overview._studies_filter(summary_df, cancer_type, location, modality)
            if summary_df.shape[0] == 0:
                class NoResultsFound(Exception):
                    pass
                raise NoResultsFound("Try Broadening the Search Criteria.")

        # Cache Search
        self.current_search = summary_df.reset_index(drop=True)

        if pretty_print:
            current_search_print = self.current_search.copy()
            pandas_pprint(current_search_print, full_cols=True, col_align='left', lift_column_width_limit=True)

        if all([collection == None, cancer_type == None, location == None, modality == None]):
            sleep(0.25)
            warn("\nSpecific search critera have not been applied.\n"
                 "If `pull()` is called, *all* collections will be downloaded.\n"
                 "Such a request could yeild several terabytes of data.\n"
                 "If you still wish to procede, consider adjusting `pull()`'s\n"
                 "`patient_limit` and `session_limit` parameters.")

    def _pull_records(self, patient_limit):
        """

        :param patient_limit:
        :return:
        """
        # Loop through and download all of the studies
        record_frames = list()
        for collection in self.current_search['Collection']:
            if self._verbose:
                print("\nDownloading records the '{0}' Collection...".format(collection))
            record_frames.append(self._Records.records_pull(collection, patient_limit=patient_limit))
        return record_frames

    def _pull_images(self, record_frames, session_limit, img_format, save_dicoms, check_cache_first):
        """

        :param record_frames:
        :param session_limit:
        :param img_format:
        :param save_dicoms:
        :param check_cache_first:
        :return:
        """
        img_frames = list()
        for records in record_frames:
            current_img_frame = self._Images.pull_img(records, session_limit, img_format, save_dicoms, check_cache_first)
            img_frames.append(current_image_frame)
        return img_frames

    def pull(self,
             patient_limit=3,
             pull_images=True,
             session_limit=1,
             img_format='png',
             save_dicoms=True,
             check_cache_first=True):
        """

        :param patient_limit:
        :param pull_images:
        :param session_limit:
        :param img_format:
        :param save_dicoms:
        :param check_cache_first:
        :return:
        """
        if self.current_search is None:
            raise AttributeError("`current_search` is empty. A search must be performed before `pull()` is called.")

        # Download Records for all of the studies
        record_frames = self._pull_records(patient_limit)

        # Download the images for all of the studies
        if pull_images:
            final_frames = self._pull_images(record_frames, session_limit, img_format, save_dicoms, check_cache_first)
        else:
            final_frames = record_frames

        # Concatenate and save
        self.pull_records = pd.concat(final_frames, ignore_index=True)

        return self.pull_records























































































































