"""

    Open-i Text Processing
    ~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
import re
import string
from itertools import chain
from collections import defaultdict

# Image Support Tools
from biovida.images.interface_support.openi._openi_support_tools import item_extract
from biovida.images.interface_support.openi._openi_support_tools import filter_unnest
from biovida.images.interface_support.openi._openi_support_tools import num_word_to_int
from biovida.images.interface_support.openi._openi_support_tools import multiple_decimal_remove

# General Support Tools
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import unescape
from biovida.support_tools.support_tools import items_null

# Data
from biovida.images.interface_support.openi._openi_imaging_modality_information import (terms_dict,
                                                                                        modality_subtypes,
                                                                                        contradictions,
                                                                                        modality_specific_subtypes)


# ----------------------------------------------------------------------------------------------------------
# Custom HTML cleaning
# ----------------------------------------------------------------------------------------------------------


def _html_text_clean(html_text, action, parse_medpix=False):
    """

    This removes HTML features commonly encountered in the 'abstract' column.
    While not perfect, it avoids use of a heavy dependency (like ``BeautifulSoup``).

    :param html_text: any HTML text
    :type html_text: ``str``
    :param action: 'entities', 'tags' or 'both'.
    :param parse_medpix:
    :type parse_medpix: ``bool``
    :return: cleaned ``html_text``
    :rtype: ``str``
    """
    if items_null(html_text) or html_text is None:
        return html_text

    # Tags to remove
    KNOWN_TAGS = ('p', 'b', 'li')

    # Remove bullet points and line breaks
    html_text = html_text.replace('&bull;', '').replace('\n', '')

    # Escape HTML entities
    if action in ('entities', 'both'):
        html_text = unescape(html_text)

    # Remove known tags
    if action in ('tags', 'both'):
        for kt in KNOWN_TAGS:
            for t in ('<{0}>'.format(kt), '</{0}>'.format(kt)):
                html_text = html_text.replace(t, ' ')

    # Limit white space to length 1 and strip() `html_text`
    html_text = cln(html_text)

    # Prettify MedPix Text
    if parse_medpix:
        for h in ('History', 'Diagnosis', 'Findings', 'Ddx', 'Dxhow'):
            html_text = html_text.replace(' {0}'.format(h), '. {0}'.format(h))
        html_text = html_text.replace('..', '.')

    return html_text


# ----------------------------------------------------------------------------------------------------------
# MedPix
# ----------------------------------------------------------------------------------------------------------


def _mexpix_info_extract(abstract):
    """

    Handles information extraction for MedPix Images

    :param abstract: a text abstract.
    :type abstract: ``str``
    :return: a dictionary with the following keys: 'Diagnosis', 'History', 'Findings', 'Ddx' and 'Dxhow'.
    :rtype: ``dict``
    """
    features = ('Diagnosis', 'History', 'Findings', 'Ddx', 'Dxhow')
    features_dict = dict.fromkeys(features, None)
    cln_abstract = cln(abstract)

    for k in features_dict:
        try:
            raw_extract = item_extract(re.findall('<p><b>' + k + ': </b>(.*?)</p><p>', cln_abstract))
            cleaned_extract = _html_text_clean(html_text=raw_extract, action='entities')
            features_dict[k] = cleaned_extract.lower() if k == 'Diagnosis' else cleaned_extract
        except:
            pass

    return features_dict


# ----------------------------------------------------------------------------------------------------------
# Patient's Sex
# ----------------------------------------------------------------------------------------------------------


def _patient_sex_extract(image_summary_info):
    """

    Tool to Extract the age of a patient.

    :param image_summary_info: some summary text of the image, e.g., 'history', 'abstract', 'image_caption'
                               or 'image_mention'. (Expected to be lower case)
    :type image_summary_info: ``str``
    :return: the sex of the patient.
    :rtype: ``str`` or ``None``
    """
    counts_dict_f = {t: image_summary_info.count(t) for t in ['female', 'woman', 'girl', ' she ', ' f ']}
    counts_dict_m = {t: image_summary_info.count(t) for t in ['male', 'man', 'boy', ' he ', ' m ']}

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

    Tool to extract the sex of the patient (female or male).

    :param history: the history of the patient.
    :type history: ``str``
    :param abstract: a text abstract.
    :type abstract: ``str``
    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param image_mention: an element from the 'image_mention' column.
    :type image_mention: ``str``
    :return: the sex of the patent.
    :rtype: ``str`` or ``None``
    """
    for source in (history, abstract, image_caption, image_mention):
        if isinstance(source, str):
            extract = _patient_sex_extract(source.lower())
            if isinstance(extract, str):
                return extract
    else:
        return None


# ----------------------------------------------------------------------------------------------------------
# Patient's Age
# ----------------------------------------------------------------------------------------------------------


def _age_refine(age_list, upper_age_bound=130):
    """

    Converts ages into floats and ages given in months to years.

    :param age_list: the list of ages evolved inside ``_patient_age_guess()``.
    :type age_list: ``list``
    :param upper_age_bound: max. valid age. Defaults to `130`.
    :type upper_age_bound: ``float`` or ``int``
    :return: a numeric age.
    :rtype: ``float`` or ``None``
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

    # Remove Invalid Ages
    valid_ages = [i for i in to_return if i <= upper_age_bound]

    # Heuristic: typically the largest value will be the age
    if len(valid_ages):
        return max(to_return)
    else:
        return None


def _patient_age_guess_abstract_clean(abstract):
    """

    Cleans the abstract for ``_patient_age_guess()``.
    This includes:

    - removing confusing references to the duration of the illness

    - numeric values from 'one' to 'one hundred and thirty' in natural language.

    :param abstract: a text abstract.
    :type abstract: ``str``
    :return: a cleaned ``abstract``
    :rtype: ``str``
    """
    # ToDo: test moving num_word_to_int() to the top of this function; d* won't capture "ten year history", for example.
    # Block: 'x year history'
    history_block = ("year history", " year history", "-year history")
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

    Extract age from a string based on it being followed by one of:

    " y", "yo ", "y.o.", "y/o", "year", "-year", " - year", " -year", "month old", " month old", "-month old",
    "months old", " months old" or "-months old".

    :param image_summary_info: some summary text of the image, e.g., 'history', 'abstract', 'image_caption'
                               or 'image_mention'.
    :type image_summary_info: ``str``
    :return: patient age.
    :rtype: ``str`` or ``None``
    """
    age_markers = (" y", "yo ", " yo ", "y.o.", "y/o", "year", "-year", " - year", " -year",
                   "month old", " month old", "-month old", "months old", " months old", "-months old")

    # Clean the input text
    cleaned_input = _patient_age_guess_abstract_clean(image_summary_info).lower()

    # Check abstract
    if len(cleaned_input):
        return filter_unnest([re.findall(r'\d+' + b, cleaned_input) for b in age_markers])
    else:
        return None


def _patient_age_guess(history, abstract, image_caption, image_mention):
    """

    Guess the age of the patient.

    :param history: history extract from the abstract of a MedPix image.
    :type history: ``str``
    :param abstract: a text abstract.
    :type abstract: ``str``
    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param image_mention: an element from the 'image_mention' column.
    :type image_mention: ``str``
    :return: patient age.
    :rtype: ``float`` or ``None``
    """
    # Loop through the inputs, search all of them for age matches
    for source in (history, abstract, image_caption, image_caption, image_mention):
        if isinstance(source, str):
            matches = _age_marker_match(source)
            if isinstance(matches, list) and len(matches):
                return _age_refine(matches)
    else:
        return None


# ----------------------------------------------------------------------------------------------------------
# Patient's Ethnicity
# ----------------------------------------------------------------------------------------------------------


def _ethnicity_guess_engine(image_summary_info):
    """

    Engine to power ``_ethnicity_guess()``.

    :param image_summary_info: some summary text of the image, e.g., 'history', 'abstract', 'image_caption' or 'image_mention'.
    :type image_summary_info: ``str``
    :return: tuple of the form ('ethnicity', 'sex'), ('ethnicity', None) or (None, None).
    :type: ``tuple``
    """
    # Init
    e_matches, s_matches = set(), set()

    # Clean the input
    image_summary_info_cln = cln(image_summary_info)

    # Define long form of references to ethnicity
    long_form_ethnicities = [('caucasian', 'white'),
                             ('black', 'african american'),
                             ('latino',),
                             ('hispanic',),
                             ('asian',),
                             ('native american',),
                             ('pacific islander', 'pacific island'),
                             ('first nations',),
                             ('aboriginal',)]

    for i in long_form_ethnicities:
        for j in i:
            if j in image_summary_info_cln.lower():
                if not ('caucasian' in e_matches and j == 'asian'):  # 'asian' is a substring in 'caucasian'.
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

    Guess the ethnicity of the patient.

    :param history: history extract from the abstract of a MedPix image.
    :type history: ``str``
    :param abstract: a text abstract.
    :type abstract: ``str``
    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param image_mention: an element from the 'image_mention' column.
    :type image_mention: ``str``
    :return: tuple of the form ('ethnicity', 'sex'), ('ethnicity', None) or (None, None).
    :rtype: ``tuple``
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


# ----------------------------------------------------------------------------------------------------------
# Patient's Disease
# ----------------------------------------------------------------------------------------------------------


def _disease_guess(title, abstract, image_caption, image_mention, list_of_diseases):
    """

    Search `title`, `abstract`, `image_caption` and `image_mention` for diseases in `list_of_diseases`

    Warning: this is likely to be confused by multiple disease names appearing in, say, ``abstract``.

    :param title:
    :param abstract:
    :param image_caption:
    :param image_mention:
    :param list_of_diseases: see ``feature_extract``.
    :type list_of_diseases: ``list``
    :return:
    :rtype: ``str`` or ``None``
    """
    # ToDo: use the first disease that appears. E.g., 'Pancreatitis in patients with diabetes' -- want 'Pancreatitis'.
    possible_diseases = list()
    for source in (title, image_caption, image_mention, abstract):
        if isinstance(source, str):
            source_clean = cln(source).lower()
            for d in list_of_diseases:
                if d in source_clean:
                    possible_diseases.append(d)
            if len(possible_diseases):  # break to prevent a later source contradicting a former one.
                break
    return possible_diseases[0] if len(possible_diseases) == 1 else None


# ----------------------------------------------------------------------------------------------------------
# Illness Duration
# ----------------------------------------------------------------------------------------------------------


def _illness_duration_guess_engine(image_summary_info):
    """

    Engine to search through the possible ways illness duration information could be represented.

    :param image_summary_info: some summary text of the image, e.g., 'history', 'abstract', 'image_caption'
                                or 'image_mention'.
    :type image_summary_info: ``str``
    :return: the best guess for the length of the illness.
    :rtype: ``float`` or ``None``.
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

    Guess the duration of an illness. Unit: years.

    :param history: history extract from the abstract of a MedPix image.
    :type history: ``str``
    :param abstract: a text abstract.
    :type abstract: ``str``
    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param image_mention: an element from the 'image_mention' column.
    :type image_mention: ``str``
    :return:
    :rtype: ``float`` or ``None``
    """
    for source in (history, abstract, image_caption, image_mention):
        if isinstance(source, str):
            duration_guess = _illness_duration_guess_engine(source.lower())
            if isinstance(duration_guess, float):
                return duration_guess
    else:
        return None


# ----------------------------------------------------------------------------------------------------------
# Imaging Technology (Modality)
# ----------------------------------------------------------------------------------------------------------


def im_scan(source):
    """

    Scan a ``source`` for modality information.

    :param source:
    :return:
    """
    # 1. Look for modality matches
    matches = defaultdict(set)
    for k, v in terms_dict.items():
        if any(i in source for i in v[0]):
            # Add that the `k` modality was found (e.g., 'mri').
            matches[k].add(None)
            # Scan `source` for subtypes (taking the optunity to
            # look for those that may not be modality specific).
            if k in modality_subtypes:
                for v2 in modality_subtypes[k]:
                    if any(j in source for j in v2):
                        matches[k].add(cln(v2[0]))

    # 2. Look for subtype matches which can be used to infer the modality itself.
    for k, v in modality_specific_subtypes.items():
        for i in v:
            if any(j in source for j in i):
                matches[k].add(cln(i[0]))

    if len(matches.keys()) != 1:
        return None
    else:
        return {k: list(filter(None, v)) for k, v in matches.items()}


def im_drop_contradictions(values):
    """

    Check for and eliminate contradictions.

    :param values:
    :return:
    """
    for c in contradictions:
        if all(i in values for i in c):
            values = [m for m in values if m not in c]
    return values


def im_formatter(dictionary):
    """

    Format the output; form: 'formal modality name: subtype1; subtype2; ...'.

    :param dictionary:
    :return:
    """
    modality = list(dictionary.keys())[0]
    modality_formal_name = terms_dict[modality][-1]
    values = im_drop_contradictions(set(dictionary[modality]))
    if not len(values):
        return modality_formal_name
    else:
        return "{0}: {1}".format(modality_formal_name, "; ".join(sorted(values)))


def _imaging_modality_guess(abstract, image_caption, image_mention):
    """

    Guess the imaging technology (modality) used to take the picture.
    Perference: ``image_caption`` > ``image_mention`` > ``abstract``.

    :param abstract: a text abstract.
    :type abstract: ``str``
    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param image_mention: an element from the 'image_mention' column.
    :type image_mention: ``str``
    :return: imaging modality.
    :rtype: ``str`` or ``None``
    """
    # Try to return by scanning `image_caption`, `image_mention` and `abstract`
    for source in (image_caption, image_mention, abstract):
        if isinstance(source, str):
            scan_rslt = im_scan(source=cln(source).lower())
            if isinstance(scan_rslt, dict) and len(scan_rslt.keys()) == 1:
                return im_formatter(scan_rslt)
    else:
        return None  # capitulate


# ----------------------------------------------------------------------------------------------------------
# Image Plane
# ----------------------------------------------------------------------------------------------------------


def _image_plane_guess(image_caption):
    """

    Guess whether the plane of the image is 'axial', 'coronal' or 'sagital'.

    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :return: see description.
    :rtype: ``str`` or ``None``
    """
    image_caption_clean = cln(image_caption).lower()
    plane_terms = [['axial', 'transverse'], ['coronal'], ['sagittal']]
    planes = [p[0] for p in plane_terms if any(i in image_caption_clean for i in p)]
    return planes[0] if len(planes) == 1 else None


# ----------------------------------------------------------------------------------------------------------
# Grids (inferred by the presence of enumerations in the text)
# ----------------------------------------------------------------------------------------------------------


def _extract_enumerations(input_str):
    """

    Extracts enumerations from strings.

    :param input_str: any string.
    :type input_str: ``str``
    :return: enumerations present in `input_str`.
    :rtype: ``list``

    :Example:

    >>> _extract_enumerations('(1a) here we see... 2. whereas here we see...')
    ...
    ['1a', '2']
    """
    # Block floating point numbers, e.g., '...there are peaks at 2.3 and 3.28.'
    # See: http://stackoverflow.com/a/4703409/4898004. (Droped second * to ignore '2.', but still catch (2.3).
    floating_point_numbers_removed = cln(re.sub(r"[-+]?\d*\.\d+", " ", input_str))

    # Futher clean the input
    cleaned_input = cln(floating_point_numbers_removed, extent=2).replace("-", "").lower()

    # Define a list to populate
    enumerations = list()

    # Markers which denote the possible presence of an enumeration.
    markers_left = ("(", "[")
    markers_right = (".", ",", ")", "]")

    # Add a marker to the end of `cleaned_input`
    cleaned_input = cleaned_input + ")" if cleaned_input[-1] not in markers_right else cleaned_input

    # Candidate enumeration
    candidate = ''

    # Walk through the cleaned string
    for i in cleaned_input:
        if i in markers_left:
            candidate = ''
        elif i not in markers_right:
            candidate += i
        elif i in markers_right:
            if len(candidate) and len(candidate) <= 2 and all(i.isalnum() for i in candidate):
                if sum(1 for c in candidate if c.isdigit()) <= 2 and sum(1 for c in candidate if c.isalpha()) <= 3:
                    enumerations.append(candidate)
            candidate = ''

    return enumerations


def _enumerations_test(l):
    """

    Test if ``l`` is an enumeration.

    :param l: a list of possible enumerations, e.g., ``['1', '2', '3']`` (assumed to be sorted and all lower case).
    :type l: ``list``
    :return: whether or not ``l`` is an enumeration.
    :rtype: ``bool``
    """
    def alnum_check(char):
        """
        Check if `char` is plausibly part of an enumeration when
        it may be part of a 'mixed' sequence, e.g., `['1', '2', '2a']`."""
        if len(char) == 1 and char.isdigit() or char.isalpha():
            return True
        elif len(char) == 2 and sum(1 for i in char if i.isdigit()) == 1 and sum(1 for i in char if i.isalpha()) == 1:
            return True
        else:
            return False

    # Check for all 'i's, e.g., ['i', 'ii', 'iii', 'iv'] etc.
    if all(item in ('i', 'ii', 'iii') for item in l[:3]):
        return True
    # Check for letter enumeration
    elif list(l) == list(string.ascii_lowercase)[:len(l)]:
        return True
    # If the above check yielded a False, ``l`` must be junk, e.g., ``['a', 'ml', 'jp']``.
    elif all(i.isalpha() for i in l):
        return False
    # Check for numeric enumeration
    elif list(l) == list(map(str, range(1, len(l) + 1))):
        return True
    # Check for a combination
    elif all(alnum_check(i) for i in l):
        return True
    else:
        return False


def _enumerations_guess(image_caption, enumerations_grid_threshold):
    """

    Look for grids based on the presence of enumerations in the image caption.

    :param image_caption:
    :type image_caption: ``str``
    :param enumerations_grid_threshold:
    :type enumerations_grid_threshold: ``int``
    :return: true if ``image_caption`` contains markers that indicate image enumeration.
    :rtype: ``bool``
    """
    image_caption_clean_lower = cln(image_caption, extent=2).lower()
    # Check for markers of enumeration like '(a-e)', '(a,e)', '(b and c)'.
    match_on = ["[(|\[][a-z]" + i + "[a-z][)|\]]" for i in ("-", ",", "and")] + ["[a-z]~[a-z]"]
    for regex in match_on:
        if len(re.findall(regex, image_caption_clean_lower)):
            return True
    else:
        # Fall back to standard enumerations
        enums = _extract_enumerations(image_caption)
        if _enumerations_test(enums) and len(enums) >= enumerations_grid_threshold:
            return True
        else:
            return False


# ----------------------------------------------------------------------------------------------------------
# Markers (inferred from the text)
# ----------------------------------------------------------------------------------------------------------


def _markers_guess(image_caption):
    """

    Currently detects the mention of 'arrows' and 'asterisks' in the image,
    based on the text.

    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :return: whether or not the image contains markers, based on the text.
    :rtype: ``bool``
    """
    features = set()

    image_caption_clean = cln(image_caption).lower()
    for term in ('arrow', 'asterisk'):
        # Look for arrows or asterisk.
        match_on = tuple(map(lambda x: x.format(term), (' {0} ', ' {0}', ' {0}s ', ' {0}s')))
        # Example: '...which are indiciated by the asterisks...'
        if any(i in image_caption_clean for i in match_on):
            features.add("{0}s".format(term))
        # Example: '...along the left side (arrows)...'
        elif len(re.findall(r'[(|\[]' + term + 's?[)|\]]', image_caption_clean)):
            features.add("{0}s".format(term))
        # Example: '...to the left (red arrows)...'
        elif len(re.findall(r'[(|\[].*? ' + term + 's?[)|\]]', image_caption_clean)):
            features.add("{0}s".format(term))

    return list(features)


# ----------------------------------------------------------------------------------------------------------
# Image Problems (Detected by Analysing their Associated Text)
# ----------------------------------------------------------------------------------------------------------


def _problematic_image_features(image_caption, enumerations_grid_threshold=2):
    """

    Currently determines whether or not the image contains
    arrows, asterisks or grids (multiple images arranged as as a grid of images).

    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param enumerations_grid_threshold: the number of enumerations required to 'believe' the image is actually a
                                        'grid' of images, e.g., '1a. here we. 1b, rather' would suffice if this
                                        parameter is equal to `2`.
    :type enumerations_grid_threshold: ``int``
    :return: information on the presence of problematic image features.
    :rtype: ``tuple``

    :Example:

    >>> _problematic_image_features(image_caption='1. left (green asterisk). 2. On the left (red arrow) we see...')
    ...
    ('arrows', 'asterisks', 'grids')
    """
    # Initialize
    features = []

    # Look for markers
    features += _markers_guess(image_caption)

    if _enumerations_guess(image_caption, enumerations_grid_threshold):
        features.append('grids')
    
    return tuple(sorted(features)) if len(features) else None


# ----------------------------------------------------------------------------------------------------------
# Outward Facing Tool
# ----------------------------------------------------------------------------------------------------------


def feature_extract(x, list_of_diseases):
    """

    Tool to extract text features from patient summaries.

    This function automatically generates the following columns when ``OpeniInterface().pull()`` is called:

    Exclusive to MedPix® Images:
        - 'history'
        - 'finding'
        - 'ddx' (differential diagnosis)
        - 'dxhow'

    For images from all sources:
        - 'age'
        - 'sex'
        - 'ethnicity'
        - 'diagnosis'
        - duration of illness ('illness_duration_years')
        - the imaging modality (e.g., MRI) used, based on the text associated with the image ('imaging_modality_from_text')
        - the plane of the image ('image_plane')
        - image problems ('arrows', 'asterisks' and 'grids') inferred from the image caption ('image_problems_from_text')

    .. warning::

        For images from sources other than *MedPix*, the ``'diagnosis'`` column may be inaccurate.


    :param x: series passed though Pandas' ``DataFrame().apply()`` method, e.g.,
              ``df.apply(lambda x: feature_extract(x, list_of_diseases), axis=1)``.

              .. note::

                   The dataframe must contain 'title', 'abstract', 'image_caption', 'image_mention'
                   and 'journal_title' columns.


    :type x: ``Pandas Series``
    :param list_of_diseases: a list of diseases (e.g., via ``DiseaseOntInterface().pull()['name'].tolist()``)
    :type list_of_diseases: ``list``
    :return: dictionary with the keys listed in the description.
    :rtype: ``dict``
    """
    # Initialize
    d = dict.fromkeys(['Diagnosis', 'History', 'Findings', 'Ddx', 'Dxhow'], None)

    if 'medpix' in x['journal_title'].lower():
        d = _mexpix_info_extract(x['abstract'])
    else:
        d['Diagnosis'] = _disease_guess(x['title'], x['abstract'], x['image_caption'],
                                        x['image_mention'], list_of_diseases)

    pairs = [('age', _patient_age_guess), ('sex', _patient_sex_guess), ('illness_duration_years', _illness_duration_guess)]
    for (k, func) in pairs:
        d[k] = func(d['History'], x['abstract'], x['image_caption'], x['image_mention'])

    # Guess the imaging technology used by using the text
    d['imaging_modality_from_text'] = _imaging_modality_guess(x['abstract'], x['image_caption'], x['image_mention'])

    # Guess image plane
    d['image_plane'] = _image_plane_guess(x['image_caption'])

    # Use the image caption to detect problems in the image
    d['image_problems_from_text'] = _problematic_image_features(x['image_caption'])

    # Guess Ethnicity
    ethnicity, eth_sex = _ethnicity_guess(d['History'], x['abstract'], x['image_caption'], x['image_mention'])
    d['ethnicity'] = ethnicity

    # Try to Extract Age from Ethnicity analysis
    if d['sex'] is None and eth_sex is not None:
        d['sex'] = eth_sex

    # Lower keys and return
    return {k.lower(): v for k, v in d.items()}
















