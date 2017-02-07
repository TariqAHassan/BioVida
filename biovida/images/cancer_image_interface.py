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
from biovida.support_tools.support_tools import combine_dicts
from biovida.support_tools.support_tools import camel_to_snake_case

cparam = CancerImgArchiveParams()
ref = cparam.cancer_img_api_ref()
dicom_m = cparam.dicom_modality_abbreviations('dict')

# r = requests.get(zip_file_url)
# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall()

study = 'ISPY1'
patient = 'ISPY1_1001'
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

    :param patient_study_df: the ``patient_study_df`` evolved inside ``baseline_records()``
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
        raise ValueError('If `patient_limit` is an intiger it must be >= 1.')

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

























