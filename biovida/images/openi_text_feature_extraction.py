"""


    Post Processing of Text Information
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
# Imports
import re
from itertools import chain

# Import tools from Open-i support tools
from biovida.images.openi_support_tools import cln
from biovida.images.openi_support_tools import item_extract
from biovida.images.openi_support_tools import filter_unnest
from biovida.images.openi_support_tools import extract_float
from biovida.images.openi_support_tools import num_word_to_int


def mexpix_info_extract(abstract):
    """

    :param abstract:
    :return:
    """
    features = ['Diagnosis', 'History', 'Findings']
    return {i: item_extract(re.findall('<p><b>' + i + ': </b>(.*?)</p><p>', cln(abstract))) for i in features}


def patient_sex_guess(abstract):
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


def age_refine(age_list):
    """

    :param age_list:
    :return:
    """
    to_return = list()
    for a in age_list:
        age = float(cln(extract_float(a)).strip())

        if 'month' in a:
            to_return.append(round(age / 12, 2))
        else:
            to_return.append(age)

    # Heuristic: typically the largest value will be the age
    return max(to_return)


def patient_age_guess(abstract):
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
    return age_refine(front_finds) if len(front_finds) else None


def feature_extract(x):
    """

    To Harvest:
        - Age
        - Sex

    :param abstract:
    :param journal:
    :return:
    """
    d = dict.fromkeys(['Diagnosis', 'History', 'Findings'], None)

    # ToDo: expand Diagnosis information harvesting to other sources.
    if 'medpix' in x['journal_title'].lower():
        d = mexpix_info_extract(x['abstract'])

    # Define string to use when trying to harvest sex and age information.
    guess_string = x['abstract'].lower() if d['History'] is None else d['History'].lower()

    # Guess Sex
    d['sex'] = patient_sex_guess(guess_string)

    # Guess Age
    d['age'] = patient_age_guess(guess_string)

    # Lower keys and return
    return {k.lower(): v for k, v in d.items()}
