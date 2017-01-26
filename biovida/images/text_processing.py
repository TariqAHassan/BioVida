"""

    Text Processing
    ~~~~~~~~~~~~~~~

"""
# ToDo:
#     1. group images by patient. Possible to do this properly?


# Imports
import re
import numpy as np
from itertools import chain

# Image Support Tools
from biovida.images._openi_support_tools import item_extract
from biovida.images._openi_support_tools import filter_unnest
from biovida.images._openi_support_tools import num_word_to_int
from biovida.images._openi_support_tools import multiple_decimal_remove

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


def _patient_sex_extract(image_summary_info):
    """

    :param image_summary_info:
    :return:
    """
    counts_dict_f = {t: image_summary_info.lower().count(t) for t in ['female', 'woman', 'girl',' f ']}
    counts_dict_m = {t: image_summary_info.lower().count(t) for t in ['male', 'man', 'boy', ' m ']}

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


def _patient_sex_guess(history, abstract, image_caption, image_mention):
    """

    Tries to extract the sex of the patient (female or male).

    :param history:
    :param abstract:
    :param image_caption:
    :param image_mention:
    :return:
    """
    for source in (history, abstract, image_caption, image_mention):
        if isinstance(source, str):
            extract = _patient_sex_extract(source)
            if isinstance(extract, str):
                return extract
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


def _patient_age_guess_abstract_clean(abstract):
    """

    Cleans the abstract for ``_patient_age_guess()``

    :param abstract:
    :return:
    """
    # Block: 'x year history'
    history_block = ["year history", " year history", "-year history"]
    hist_matches = [re.findall(r'\d*\.?\d+' + drop, abstract) for drop in history_block]

    # Clean and recompose the string
    cleaned_abstract = cln(" ".join([abstract.replace(r, "") for r in chain(*filter(None, hist_matches))])).strip()
    hist_matches_flat = filter_unnest(hist_matches)

    if len(hist_matches_flat):
        return num_word_to_int(" ".join([abstract.replace(r, "") for r in hist_matches_flat])).strip()
    else:
        return num_word_to_int(abstract)


def _age_marker_match(image_summary_info):
    """

    :param image_summary_info:
    :return:
    """
    age_markers = [" y", "yo ", "y.o.", "y/o", "year", "-year", " - year", " -year",
                   "month old", " month old", "-month old", "months old", " months old", "-months old"]

    # Clean the input text
    cleaned_input = _patient_age_guess_abstract_clean(image_summary_info)

    # Check abstract
    if len(cleaned_input):
        return filter_unnest([re.findall(r'\d+' + b, cleaned_input) for b in age_markers])
    else:
        return None


def _patient_age_guess(history, abstract, image_caption, image_mention):
    """

    :param abstract:
    :param image_caption:
    :param image_mention:
    :param History:
    :return:
    """
    # Loop through the inputs, search all of them for age matches
    for source in (history, abstract, image_caption, image_caption):
        if isinstance(source, str):
            matches = _age_marker_match(source)
            if isinstance(matches, list) and len(matches):
                return _age_refine(matches)
    else:
        return None


def _imaging_technology_guess(abstract, image_caption, image_mention):
    """

    :param abstract:
    :param image_caption:
    :param image_mention:
    :return:
    """
    # {abbreviated name, [alternative names]}
    terms_dict = {"ct": ['computed topography'],
                  "mri": ['magnetic resonance imaging'],
                  "x-ray": ['xray'],
                  "ultrasound": []}

    # Loop though and look for matches
    matches = set()
    for source in (abstract, image_caption, image_mention):
        if isinstance(source, str):
            for k, v in terms_dict.items():
                if k in source.lower() or any(i in source.lower() for i in v):
                    matches.add(k)

    return list(matches)[0] if len(matches) == 1 else None


def _illness_duration_guess_engine(image_summary_info):
    """

    :param image_summary_info:
    :return:
    """
    cleaned_source = cln(image_summary_info.replace("-", " "), extent=1)

    match_terms = [('month history of', 'm'), ('year history of', 'y')]
    hist_matches = [(re.findall(r"\d*\.?\d+ (?=" + t + ")", cleaned_source), u) for (t, u) in match_terms]

    def time_adj(mt):
        if len(mt[0]) == 1:
            cleaned_date = re.sub("[^0-9.]", "", cln(mt[0][0], extent=2)).strip()
            if cleaned_date.count(".") > 1:
                return None
            if mt[1] == 'y':
                return float(cleaned_date)
            elif mt[1] == 'm':
                return float(cleaned_date) / 12.0
        else:
            return None

    # Filter invalid extracts
    durations = list(filter(None, map(time_adj, hist_matches)))

    return durations[0] if len(durations) == 1 else None


def _illness_duration_guess(history, abstract, image_caption, image_mention):
    """

    :param history:
    :param abstract:
    :param image_caption:
    :param image_mention:
    :return:
    """
    for source in (history, abstract, image_caption, image_mention):
        if isinstance(source, str):
            duration_guess = _illness_duration_guess_engine(source)
            if isinstance(duration_guess, float):
                return duration_guess
    else:
        return None


def _ethnicity_guess_engine(image_summary_info):
    """

    :param image_summary_info:
    :return:
    """
    # Init
    e_matches, s_matches = set(), set()

    # Clean the input
    image_summary_info_cln = cln(image_summary_info)
    image_summary_info_cln_lwr = image_summary_info_cln.lower()

    # Define long form of references to ethnicity
    long_form_ethnicities = [('caucasian', 'white'),
                             ('black', 'african american'),
                             ('latino',),
                             ('hispanic',),
                             ('asian',),
                             ('native american',),
                             ('first nations',)]

    for i in long_form_ethnicities:
        for j in i:
            if j in image_summary_info_cln:
                e_matches.add(i[0])

    # Define short form of references to ethnicity
    short_form_ethnicities = [(' AM ', 'asian', 'male'), (' AF ', 'asian', 'female'),
                              (' BM ', 'black', 'male'), (' BF ', 'black', 'female'),
                              (' WM ', 'caucasian', 'male'), (' WF ', 'caucasian', 'female')]

    for (abbrev, eth, sex) in short_form_ethnicities:
        if abbrev in image_summary_info_cln:
            e_matches.add(eth)
            s_matches.add(sex)

    # Render output
    patient_ethnicity = list(e_matches)[0] if len(e_matches) == 1 else None
    patient_sex = list(s_matches)[0] if len(s_matches) == 1 else None

    return patient_ethnicity, patient_sex


def _ethnicity_guess(history, abstract, image_caption, image_mention):
    """

    :param history:
    :param abstract:
    :param image_caption:
    :param image_mention:
    :return:
    """
    matches = list()
    for source in (history, abstract, image_caption, image_mention):
        if isinstance(source, str):
            matches.append(_ethnicity_guess_engine(source))

    # 1. Prefer both pieces of information
    both_matches = [i for i in matches if not all(j is None for j in i)]
    if len(both_matches):
        return both_matches[0]  # simply use the first one

    # 2. Search for single matches
    single_matches = [i for i in matches if any(j is not None for j in i)]
    if len(single_matches):
        return single_matches[0]  # again, simply use the first one
    else:
        return None, None


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

    :param x: series passed though Pandas' `DataFrame().apply()` method, e.g.,
              ``df.apply(feature_extract, axis=1)``. The dataframe must contain
              'abstract' and 'journal_title' columns.
    :type x: ``Pandas Series``
    :return: dictionary with the following keys: 'diagnosis', 'history', 'findings', 'sex' and 'age'.
    :rtype: ``dict``
    """
    d = dict.fromkeys(['Diagnosis', 'History', 'Findings'], None)

    # ToDo: expand Diagnosis information harvesting to other sources.
    if 'medpix' in x['journal_title'].lower():
        d = _mexpix_info_extract(x['abstract'])

    # Guess Age
    d['age'] = _patient_age_guess(d['History'], x['abstract'], x['image_caption'], x['image_mention'])

    # Guess Sex
    d['sex'] = _patient_sex_guess(d['History'], x['abstract'], x['image_caption'], x['image_mention'])

    # Guess illness duration
    d['illness_duration'] = _illness_duration_guess(d['History'], x['abstract'], x['image_caption'], x['image_mention'])

    # Guess the imaging technology used
    d['caption_imaging_tech'] = _imaging_technology_guess(x['abstract'], x['image_caption'], x['image_mention'])

    # Guess Ethnicity
    ethnicity, eth_sex = _ethnicity_guess(d['History'], x['abstract'], x['image_caption'], x['image_mention'])
    d['ethnicity'] = ethnicity

    # Try to Extract Age from Ethnicity analysis
    if d['sex'] is None and eth_sex is not None:
        d['sex'] = eth_sex

    # Lower keys and return
    return {k.lower(): v for k, v in d.items()}






































