"""

    The Cancer Imaging Archive Interface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import io
import os
import pickle
import zipfile
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

from biovida.images.ci_api_key import API_KEY

from biovida.support_tools._cache_management import package_cache_creator
from biovida.images._resources.cancer_image_parameters import CancerImgArchiveParams

from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import only_numeric
from biovida.support_tools.support_tools import combine_dicts

cparam = CancerImgArchiveParams()
ref = cparam.cancer_img_api_ref()
dicom_m = cparam.dicom_modality_abbreviations('dict')

# r = requests.get(zip_file_url)
# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall()


# ---------------------------------------------------------------------------------------------
# Summarize Studies Provided Through The Cancer Imaging Archive
# ---------------------------------------------------------------------------------------------


class CancerImgArchiveOverview(object):
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
    def __init__(self, verbose=False, cache_path=None, tcia_homepage='http://www.cancerimagingarchive.net'):
        self._verbose = verbose
        self._tcia_homepage = tcia_homepage
        _, self._created_img_dirs = package_cache_creator(sub_dir='images', cache_path=cache_path, to_create=['aux'])

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
            lambda x: [dicom_m.get(cln(i), i) for i in cln(x).split(", ")])

        # Parse the Location Column (account for special case: 'Head-Neck').
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

    def studies(self, collection=None, cancer_type=None, location=None, download_override=False):
        """

        Method to Search for studies on The Cancer Imaging Archive.

        :param collection: a collection (study) hosted by The Cancer Imaging Archive.
        :type collection: ``str`` or ``None``
        :param cancer_type: a string or list/tuple of specifying cancer types.
        :type cancer_type: ``str``, ``iterable`` or ``None``
        :param location: a string or list/tuple of specifying body locations.
        :type location: ``str``, ``iterable`` or ``None``
        :param download_override: If ``True``, override any existing database currently cached and download a new one.
                                  Defaults to ``False``.
        :return: a dataframe containing the search results.
        :rtype: ``Pandas DataFrame``

        :Example:

        >>> CancerImgArchiveOverview().studies(cancer_type=['Squamous'], location=['head', 'neck'])
        ...
           Collection               CancerType               Modalities  Subjects     Location    Metadata  ...
        0  TCGA-HNSC  Head and Neck Squamous Cell Carcinoma  CT, MR, PT     164     [Head, Neck]    Yes     ...

        """
        # Load the Summary Table
        summary_df = self._all_studies_cache_mngt(download_override)

        # Filter by Collection
        if isinstance(collection, str) and any(i is not None for i in (cancer_type, location)):
            raise ValueError("Both `cancer_types` and `location` must be ``None`` if a `collection` name is passed.")
        elif isinstance(collection, str):
            summary_df = summary_df[summary_df['Collection'].str.strip().str.lower() == collection.strip().lower()]
            if summary_df.shape[0] == 0:
                raise AttributeError("No Collection with the name '{0}' could be found.".format(collection))
            else:
                return summary_df

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

        if summary_df.shape[0] == 0:
            raise AttributeError("No Results Found. Try Broading the Search Criteria.")
        else:
            return summary_df.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Pull Records from The Cancer Imaging Archive
# ---------------------------------------------------------------------------------------------


study = 'ISPY1'
root_url = 'https://services.cancerimagingarchive.net/services/v3/TCIA'

# 1. Pick a Study
# 2. Download all the patients in that study
# 3. Make API calls as the program loops though getSeries' queries.
# 4. patient_limit to baseline images (from StudyInstanceUID)


def _extract_study(study):
    """

    Download all patients in a given study.

    :param study:
    :return:
    """
    url = '{0}/query/getPatientStudy?Collection={1}&format=csv&api_key={2}'.format(root_url, study, API_KEY)
    return pd.DataFrame.from_csv(url).reset_index()


def _date_index_map(list_of_dates):
    """

    Returns a dict of the form: ``{date: index in ``list_of_dates``, ...}``

    :param list_of_dates:
    :return:
    """
    return {k: i for i, k in enumerate(sorted(list_of_dates), start=1)}


def _summarize_study_by_patient(study):
    """

    Summarizes a study by patient.
    Note: patient_limits summary to baseline (i.e., follow ups are excluded).

    :param study:
    :return: nested dictionary of the form:

            ``{PatientID: {StudyInstanceUID: {'sex':..., 'age': ..., 'stage': ..., 'StudyDate': ...}}}``

    :rtype: ``dict``
    """
    # Download a summary of all patients in a study
    study_df = _extract_study(study)

    # Convert StudyDate to datetime
    study_df['StudyDate'] = pd.to_datetime(study_df['StudyDate'], infer_datetime_format=True)

    # Divide Study into stages (e.g., Baseline (stage 1); Baseline + 1 Month (stage 2), etc.
    stages = study_df.groupby('PatientID').apply(lambda x: _date_index_map(x['StudyDate'].tolist())).to_dict()

    # Apply stages
    study_df['Stage'] = study_df.apply(lambda x: stages[x['PatientID']][x['StudyDate']], axis=1)

    # Define Columns to Extract from study_df
    valuable_cols = ('PatientID', 'StudyInstanceUID', 'Stage', 'PatientSex', 'PatientAge', 'StudyDate')

    # Convert to a nested dictionary
    patient_dict = dict()
    for pid, si_uid, stage, sex, age, date in zip(*[study_df[c] for c in valuable_cols]):
        inner_nest = {'sex': sex, 'age': age, 'stage': stage, 'StudyDate': date}
        if pid not in patient_dict:
            patient_dict[pid] = {si_uid: inner_nest}
        else:
            patient_dict[pid] = combine_dicts(patient_dict[pid], {si_uid: inner_nest})

    return patient_dict


def _patient_img_summary(patient, patient_dict):
    """

    Harvests the Cancer Image Archive's Text Record of all baseline images for a given patient
    in a given study.

    :param patient:
    :return:
    """
    # Select an individual Patient
    url = '{0}/query/getSeries?Collection=ISPY1&PatientID={1}&format=csv&api_key={2}'.format(
        root_url, patient, API_KEY)
    patient_df = pd.DataFrame.from_csv(url).reset_index()

    def upper_first(s):
        return "{0}{1}".format(s[0].upper(), s[1:])

    # Add Sex, Age, Stage, and StudyDate
    patient_info = patient_df['StudyInstanceUID'].map(
        lambda x: {upper_first(k): patient_dict[x][k] for k in ('sex', 'age', 'stage', 'StudyDate')})
    patient_df = patient_df.join(pd.DataFrame(patient_info.tolist()))

    # Add PatientID
    patient_df['PatientID'] = patient

    return patient_df


def _clean_patient_study_df(patient_study_df):
    """

    Cleans the input in the following ways:

    - convert 'F' --> 'Female' and 'M' --> 'Male'

    - Converts the 'Age' column to numeric (years)

    - Remove line breaks in the 'ProtocolName' and 'SeriesDescription' columns

    - Add Full name for modaility (ModailityFull)

    - Convert the 'SeriesDate' column to datetime

    :param patient_study_df: the ``patient_study_df`` evolved inside ``study_records()``
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

    # Remove uneeded line break marker
    for c in ('ProtocolName', 'SeriesDescription'):
        patient_study_df[c] = patient_study_df[c].map(lambda x: cln(x.replace("\/", " ")), na_action='ignore')

    # Add the full name for modality.
    patient_study_df['ModalityFull'] = patient_study_df['Modality'].map(
        lambda x: dicom_m.get(x, np.NaN), na_action='ignore'
    )

    # Convert SeriesDate to datetime
    patient_study_df['SeriesDate'] = pd.to_datetime(patient_study_df['SeriesDate'], infer_datetime_format=True)

    # Sort and Return
    return patient_study_df.sort_values(by=['PatientID', 'Stage'])


def study_records(study, patient_limit=3):
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
    # Summarize a study by patient
    study_dict = _summarize_study_by_patient(study)

    # Check for invalid `patient_limit` values:
    if not isinstance(patient_limit, int) and patient_limit is not None:
        raise ValueError('`patient_limit` must be an intiger or `None`.')
    elif isinstance(patient_limit, int) and patient_limit < 1:
        raise ValueError('If `patient_limit` is an intiger it must be greater than or equal to 1.')

    # Define number of patients to extract
    s_patients = sorted(study_dict.keys())
    patients_to_obtain = s_patients[:patient_limit] if isinstance(patient_limit, int) else s_patients

    # Evolve a dataframe ('frame') for the baseline images of all patients
    frames = list()
    for patient in tqdm(patients_to_obtain):
        frames.append(_patient_img_summary(patient, patient_dict=study_dict[patient]))

    # Concatenate baselines frame for each patient
    patient_study_df = pd.concat(frames, ignore_index=True)

    # Add Study name
    patient_study_df['StudyName'] = study

    # Clean the dataframe and return
    return _clean_patient_study_df(patient_study_df)

























