"""

    The Cancer Imaging Archive Interface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import io
import pickle
import zipfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from biovida.images.ci_api_key import API_KEY

from biovida.support_tools._cache_management import package_cache_creator
from biovida.images._resources.cancer_image_parameters import CancerImgArchiveParams
from biovida.support_tools.printing import pandas_pprint

from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import only_numeric
from biovida.support_tools.support_tools import camel_to_snake_case

cparam = CancerImgArchiveParams()
ref = cparam.cancer_img_api_ref()
dicom_m = cparam.dicom_modality_abbreviations('dict')

# r = requests.get(zip_file_url)
# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall()

root_url = 'https://services.cancerimagingarchive.net/services/v3/TCIA'

# 1. Pick a Study
# 2. Download all the patients in that study
# 3. Make API calls as the program loops though getSeries' queries.
# 4. Limit to baseline images (from StudyInstanceUID)


def _extract_study(study):
    """

    Download all patients in a given study.

    :param study:
    :return:
    """
    url = '{0}/query/getPatientStudy?Collection={1}&format=csv&api_key={2}'.format(root_url, study, API_KEY)
    return pd.DataFrame.from_csv(url).reset_index()


def _summarize_study_by_patient(study):
    """

    Summarizes a study by patient.
    Note: Limits summary to baseline (i.e., follow ups are excluded).

    :param study:
    :return:
    """
    # Download a summary of all patients in a study
    study_df = _extract_study(study)

    # Convert StudyDate to datetime
    study_df['StudyDate'] = pd.to_datetime(study_df['StudyDate'], infer_datetime_format=True)

    # Get Unique PatientIDs
    baseline_date = study_df.groupby('PatientID').apply(lambda x: min(x['StudyDate'].tolist())).to_dict()

    # Get StudyInstanceUID which correspondings to the baseline date
    study_df['baseline'] = study_df.apply(lambda x: baseline_date[x['PatientID']] == x['StudyDate'], axis=1)

    # Drop Non-baseline entries
    study_df = study_df[study_df['baseline'] == True].reset_index(drop=True).drop('baseline', axis=1)

    # Mapping of PatientIDs and baseline StudyInstanceUID.
    baseline_study_uids = dict(zip(study_df['PatientID'], study_df['StudyInstanceUID']))

    # Convert the valuable parts of study_df into a dict.
    study_df_cols = zip(*[study_df[c] for c in ('PatientID', 'PatientSex', 'PatientAge', 'StudyDate', 'StudyInstanceUID')])

    # Return a summary by patient
    return {p: {'sex': s, 'age': a, 'date': d, 'si_UID': si} for p, s, a, d, si in study_df_cols}


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

    # Filter to baseline
    patient_df = patient_df[patient_df['StudyInstanceUID'] == patient_dict['si_UID']].reset_index(drop=True)

    # Add Sex, Age, StudyDate and PatientID
    patient_df['Sex'] =patient_dict['sex']
    patient_df['Age'] = patient_dict['age']
    patient_df['StudyDate'] = patient_dict['date']
    patient_df['PatientID'] = patient

    return patient_df


def _clean_baseline_df(baseline_df):
    """

    Cleans the input in the following ways:

    - convert 'F' --> 'Female' and 'M' --> 'Male'

    - Converts the 'Age' column to numeric (years)

    - Remove line breaks in the 'ProtocolName' and 'SeriesDescription' columns

    - Add Full name for modaility (ModailityFull)

    - Convert the 'SeriesDate' column to datetime

    :param baseline_df: the ``baseline_df`` evolved inside ``baseline_records()``
    :type baseline_df: ``Pandas DataFrame``
    :return: a cleaned ``baseline_df``
    :rtype: ``Pandas DataFrame``
    """
    # convert 'F' --> 'female' and 'M' --> 'male'.
    baseline_df['Sex'] = baseline_df['Sex'].map(
        lambda x: {'F': 'female', 'M': 'male'}.get(cln(str(x)).upper(), x), na_action='ignore')

    # Convert entries in the 'Age' Column to floats.
    baseline_df['Age'] = baseline_df['Age'].map(
        lambda x: only_numeric(x) / 12.0 if 'M' in str(x).upper() else only_numeric(x), na_action='ignore')

    # Remove uneeded line break marker
    for c in ('ProtocolName', 'SeriesDescription'):
        baseline_df[c] = baseline_df[c].map(lambda x: cln(x.replace("\/", " ")), na_action='ignore')

    # Add the full name for modality.
    baseline_df['ModalityFull'] = baseline_df['Modality'].map(lambda x: dicom_m.get(x, np.NaN), na_action='ignore')

    # Convert SeriesDate to datetime
    baseline_df['SeriesDate'] = pd.to_datetime(baseline_df['SeriesDate'], infer_datetime_format=True)

    return baseline_df


def baseline_records(study, limit=3):
    """
    
    Extract all baselines images for all patients in a given study.
    
    :param study:
    :type study: ``str``
    :param limit: limit on the number of patients to extract. Patient IDs are sorted prior to this limit being imposed.
                  if ``None``, no limit will be imposed. Defaults to `3`.
    :type limit: ``int`` or ``None``
    :return: a dataframe of all baseline images
    :rtype: ``Pandas DataFrame``
    """
    # Summarize a study by patient
    study_dict = _summarize_study_by_patient(study)

    # Check for invalid `limit` values:
    if not isinstance(limit, int) and limit is not None:
        raise ValueError('`limit` must be an intiger or `None`.')
    elif isinstance(limit, int) and limit < 1:
        raise ValueError('If `limit` is an intiger it must be >= 1.')

    # Define number of patients to extract
    patients_to_obtain = sorted(study_dict.keys())[:limit] if isinstance(limit, int) else sorted(study_dict.keys())

    # Evolve a dataframe ('frame') for the baseline images of all patients
    frames = list()
    for patient in tqdm(patients_to_obtain):
        frames.append(_patient_img_summary(patient, study_dict[patient]))

    # Concatenate baselines frame for each patient
    baseline_df = pd.concat(frames, ignore_index=True)

    # Clean the dataframe and return
    return _clean_baseline_df(baseline_df)


































