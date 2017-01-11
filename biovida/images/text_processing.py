"""

    Text Processing
    ~~~~~~~~~~~~~~~

"""
# ToDo:
#     1. search abstract for diagnostic information.
#     2. search the 'problems' column for diagnosis information.
#     3. mine abstract for imaging_tech (or 'image_caption' column)
#     4. group images by patient
#     5. add ethnicity feature extraction
#     6. handle ethnicity and sex extraction (e.g., WF = white, female)
#     7. add extraction of illness length (e.g., 'x month history of...')

# Imports
import re
import numpy as np
from itertools import chain

# Image Support Tools
from biovida.images.openi_support_tools import item_extract
from biovida.images.openi_support_tools import filter_unnest
from biovida.images.openi_support_tools import num_word_to_int
from biovida.images.openi_support_tools import multiple_decimal_remove

# General Support Tools
from biovida.support_tools.support_tools import cln


def _mexpix_info_extract(abstract):
    """

    :param abstract:
    :return:
    """
    features = ['Diagnosis', 'History', 'Findings']
    features_dict = dict.fromkeys(features, None)
    cln_abstract = cln(abstract)

    for k in features_dict:
        try:
            features_dict[k] = item_extract(re.findall('<p><b>' + k + ': </b>(.*?)</p><p>', cln_abstract))
        except:
            pass

    return features_dict


def _patient_sex_guess(abstract):
    """

    Tries to extract the sex of the patient (female or male).

    :param abstract:
    :return:
    """
    counts_dict_f = {t: abstract.lower().count(t) for t in ['female', 'woman', 'girl',' f ']}
    counts_dict_m = {t: abstract.lower().count(t) for t in ['male', 'man', 'boy', ' m ']}

    # Block Conflicting Information
    if sum(counts_dict_f.values()) > 0 and sum([v for k, v in counts_dict_m.items() if k not in ['male', 'man']]) > 0:
        return None

    # Check for sex information
    if any(x > 0 for x in counts_dict_f.values()):
        return 'female'
    elif any(x > 0 for x in counts_dict_m.values()):
        return 'male'
    else:
        return None


def _age_refine(age_list):
    """

    :param age_list:
    :return:
    """
    to_return = list()
    for a in age_list:
        age = multiple_decimal_remove(a)
        if age is None:
            return None
        try:
            if 'month' in a:
                to_return.append(round(float(age) / 12, 2))
            else:
                to_return.append(float(age))
        except:
            return None

    # Heuristic: typically the largest value will be the age
    return max(to_return)


def _patient_age_guess(abstract):
    """

    Forms:
        - x yo
        - x year
        - x-year
        - DOB: x
        - DOB x

    Note: should also consider 'elderly' if no other information can be harvested

    :param abstract:
    :return:
    """
    back = [" y", "yo ", "y.o.", "y/o", "year", "-year", " - year", " -year",
            "month old", " month old", "-month old", "months old", " months old", "-months old"]
    # front = ["dob: ", "dob "]

    # Block: 'x year history'
    history_block = ["year history", " year history", "-year history"]
    hist_matches = [re.findall(r'\d*\.?\d+' + drop, abstract) for drop in history_block]

    # Clean and recompose the string
    cleaned_abstract = cln(" ".join([abstract.replace(r, "") for r in chain(*filter(None, hist_matches))])).strip()
    hist_matches_flat = filter_unnest(hist_matches)

    if len(hist_matches_flat):
        cleaned_abstract = num_word_to_int(" ".join([abstract.replace(r, "") for r in hist_matches_flat])).strip()
    else:
        cleaned_abstract = num_word_to_int(abstract)

    # Block processing of empty strings
    if not len(cleaned_abstract):
        return None

    # Try back
    front_finds = filter_unnest([re.findall(r'\d+' + b, cleaned_abstract) for b in back])

    # Return
    return _age_refine(front_finds) if len(front_finds) else None


def feature_extract(x):
    """

    Tool to extract text features from patient summaries.

    If a MedPix® Image:
        - diagnosis
        - history
        - finding

    For images from all sources:
        - sex
        - age

    :param x: Pandas Seris passed though pandas `DataFrame().apply()` method, e.g.,
              ``df.apply(feature_extract, axis=1)``. Must contain 'abstract' and 'journal_title' columns.
    :type x: ``Pandas Series``
    :return: dictionary with the following keys: 'diagnosis', 'history', 'findings', 'sex' and 'age'.
    :rtype: ``dict``
    """
    d = dict.fromkeys(['Diagnosis', 'History', 'Findings'], None)

    # ToDo: expand Diagnosis information harvesting to other sources.
    if 'medpix' in x['journal_title'].lower():
        d = _mexpix_info_extract(x['abstract'])

    # Define string to use when trying to harvest sex and age information.
    if not isinstance(x['abstract'], str):
        guess_string = None
    elif d['History'] is None:
        guess_string = x['abstract'].lower()
    else:
        guess_string = d['History'].lower()

    # Guess Sex
    d['sex'] = _patient_sex_guess(guess_string) if isinstance(guess_string, str) else np.NaN

    # Guess Age
    d['age'] = _patient_age_guess(guess_string) if isinstance(guess_string, str) else np.NaN

    # Lower keys and return
    return {k.lower(): v for k, v in d.items()}


















