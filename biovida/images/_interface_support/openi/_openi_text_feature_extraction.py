# coding: utf-8

"""

    Open-i Text Feature Extraction
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import re
import string
from bs4 import BeautifulSoup
from collections import defaultdict

# Image Support Tools
from biovida.images._interface_support.openi.openi_support_tools import (filter_unnest,
                                                                         num_word_to_int,
                                                                         multiple_decimal_remove)

# General Support Tools
from biovida.support_tools.support_tools import (cln,
                                                 unescape,
                                                 order_set,
                                                 multi_replace,
                                                 remove_html_bullet_points)

# Data
from biovida.images._interface_support.openi._openi_imaging_modality_information import (terms_dict,
                                                                                         modality_subtypes,
                                                                                         contradictions,
                                                                                         modality_specific_subtypes)


CLINICAL_ARTICLE_TYPES = ('encounter', 'case_report', 'radiology_report')


# ----------------------------------------------------------------------------------------------------------
# Abstract Processing
# ----------------------------------------------------------------------------------------------------------


def _key_clean(key):
    """

    Cleans the key for the ``parsed_abstract`` dictionary inside ``_abstract_parser()``.

    :param key: ``key`` as evolved inside ``_abstract_parser()``.
    :type key: ``str``
    :return: see description.
    :rtype: ``str``
    """
    key_lowered = key.lower()
    if 'history' in key_lowered or 'background' in key_lowered:
        return 'background'
    if 'material' in key_lowered or 'method' in key_lowered:
        return "methods"
    if 'finding' in key_lowered or 'result' in key_lowered:
        return 'results'
    else:
        return cln(cln(key).replace(":", "")).replace("/", "_").replace(" ", "_").lower()


def _value_clean(value):
    """

    Cleans the value for the ``parsed_abstract`` dictionary inside ``_abstract_parser()``.

    :param value: ``value`` as evolved inside ``_abstract_parser()``.
    :type value: ``str``
    :return: see description.
    :rtype: ``str``
    """
    nans = ('na', 'n/a', 'nan', 'none', '')
    value = cln(unescape(value).strip(";"))
    return value if value.lower() not in nans else None


def _abstract_parser(abstract):
    """

    Take a raw HTML abstract and generate dictionary
    using the items in bold (e.g., '<b>History: </b>') as the keys
    and the result of the information inside the <p>...</p> tags as the value.

    :param abstract: a text abstract.
    :type abstract: ``str``
    :return: a dictionary of the form described above.
    :rtype: ``None`` or ``dict``

    :Example:

    >>> _abstract_parser(abstract='<p><b>History: </b>The patient presented with xyz.</p>')
    ...
    {'history': 'The patient presented with xyz.'}

    """
    if not isinstance(abstract, str):
        return None

    soup = BeautifulSoup(remove_html_bullet_points(abstract), 'lxml')

    parsed_abstract = dict()
    for p in soup.find_all('p'):
        # Look for Key
        key_candidate = p.find_all('b')
        if len(key_candidate) == 1:
            key = _key_clean(key=key_candidate[0].text)
            if len(key):
                # Look for Value
                contents = p.contents
                if len(contents) == 2:
                    parsed_abstract[key] = _value_clean(value=contents[-1])

    return parsed_abstract if parsed_abstract else None


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


def _patient_sex_guess(background, abstract, image_caption, image_mention):
    """

    Tool to extract the sex of the patient (female or male).

    :param background: the background of the patient/study.
    :type background: ``str``
    :param abstract: a text abstract.
    :type abstract: ``str``
    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param image_mention: an element from the 'image_mention' column.
    :type image_mention: ``str``
    :return: the sex of the patent.
    :rtype: ``str`` or ``None``
    """
    for source in (background, abstract, image_caption, image_mention):
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
    # E.g., '...eight year history of...' --> '...8 year history of...'
    abstract_int = num_word_to_int(abstract)

    # Block: 'x year history'
    history_block = ("year history", " year history", "-year history")

    # Flatten list comp.
    hist_matches = filter_unnest([re.findall(r'\d*\.?\d+' + drop, abstract_int) for drop in history_block])

    if len(hist_matches):
        return cln(" ".join([abstract_int.replace(r, "") for r in hist_matches]))
    else:
        return cln(abstract_int)


def _age_marker_match(image_summary_info):
    """

    Extract age from a string based on it being followed by one of
    those in ``age_markers``.

    Note: will be confused by strings like: 'The first six years of substantial'...

    :param image_summary_info: some summary text of the image, e.g., 'history', 'abstract', 'image_caption'
                               or 'image_mention'.
    :type image_summary_info: ``str``
    :return: patient age.
    :rtype: ``str`` or ``None``
    """
    # Note: given " y", it is not clear how much help something like " yo" is, given the former
    # will already catch it.
    age_markers = (" y", "yo ", " yo ", "y.o.", "y/o", "year", "-year", " -year", " ans ",
                   "month old", " month old", "-month old", "months old", " months old", "-months old")

    # Clean the input text
    cleaned_input = _patient_age_guess_abstract_clean(image_summary_info).replace(" - ", "-").lower()

    # Check abstract
    if len(cleaned_input):
        rslts = filter_unnest([re.findall(r'\d*\.\d+' + b + '|\d+' + b, cleaned_input) for b in age_markers])
        return rslts if len(rslts) else None
    else:
        return None


def _patient_age_guess(background, abstract, image_caption, image_mention):
    """

    Guess the age of the patient.

    :param background: the background of the patient/study.
    :type background: ``str``
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
    for source in (background, abstract, image_caption, image_mention):
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
    e_matches, s_matches = set(), set()

    # Clean the input
    image_summary_info_cln = cln(image_summary_info)

    def eth(ethnicity):
        return ["{0} {1}".format(ethnicity, s) for s in ['adult', 'male', 'female']]

    # Define long form of references to ethnicity
    long_form_ethnicities = {
        'caucasian': ['caucasian'] + eth('white'),
        'black': ['african american'] + eth('black'),
        'latino': ['latino'],
        'asian': ['asian'],
        'native american': ['native american'],
        'pacific islander': ['pacific islander'],
        'first nations': ['first nations'],
        'aboriginal': ['aboriginal']
    }

    for k, v in long_form_ethnicities.items():
        for j in v:
            if j in image_summary_info_cln.lower():
                if not ('caucasian' in e_matches and j == 'asian'):  # 'asian' is a substring in 'caucasian'.
                    e_matches.add(k)

    # Define short form of references to ethnicity. ToDo: use regex to anchor text.
    short_form_ethnicities = [(' BM ', 'black', 'male'), (' BF ', 'black', 'female'),
                              (' WM ', 'caucasian', 'male'), (' WF ', 'caucasian', 'female')]

    for (abbrev, eth, sex) in short_form_ethnicities:
        if abbrev in image_summary_info_cln:
            e_matches.add(eth)
            s_matches.add(sex)

    patient_ethnicity = list(e_matches)[0] if len(e_matches) == 1 else None
    patient_sex = list(s_matches)[0] if len(s_matches) == 1 else None

    return patient_ethnicity, patient_sex


def _ethnicity_guess(background, abstract, image_caption, image_mention):
    """

    Guess the ethnicity of the patient.

    :param background: the background of the patient/study.
    :type background: ``str``
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
    for source in (background, abstract, image_caption, image_mention):
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


def _remove_substrings(list_of_terms):
    """

    Remove list elements that are substrings of other list elements.

    :param list_of_terms: any list of strings and their starting position in the source string.
    :type list_of_terms: ``iterable``
    :return: see description.
    :type: ``list_of_terms`` with any substrings of other terms removed.

    :Example:

    # Source string: "The patient presented with congenital heart disease...".
    >>> remove_substrings(list_of_terms=[[38, 'heart disease'], [27, 'congenital heart disease'], [44, 'disease']])
    ...
    [27, 'congenital heart disease']

    """
    if len(list_of_terms) <= 1:
        return list_of_terms
    else:
        rslt = list()
        for i in list_of_terms:
            if not any(i[-1] in j[-1] for j in list_of_terms if j[-1] != i[-1]):
                rslt.append(i)
        return rslt


def _disease_guess(problems, title, background, abstract, image_caption, image_mention, list_of_diseases):
    """

    Search `problems`, `title`, `abstract`, `image_caption` and `image_mention` for diseases in `list_of_diseases`

    Near future: count the number of times the disease is mentioned in all sources and use that to guide decision...

    Future Directions:

        Shift to more advanced approaches and consider using one or all of:

        - the NegEx Algorithm (https://healthinformatics.wikispaces.com/NegEx+Algorithm).
        - pyConTextNLP library (https://github.com/chapmanbe/pyConTextNLP) -- more recent tool than NegEx.
        - consider using the 'image_caption_concepts' and mesh column.

    Also see:
        - Interleaved Text/Image Deep Mining on a Large-Scale Radiology Database for Automated Image Interpretation
          (https://arxiv.org/abs/1505.00670).
        - https://irp.nih.gov/blog/post/2016/11/challenges-to-training-artificial-intelligence-with-medical-imaging-data

    :param problems:
    :param title:
    :param background:
    :param abstract:
    :param image_caption:
    :param image_mention:
    :param list_of_diseases: see ``feature_extract``.
    :type list_of_diseases: ``list``
    :return:
    :rtype: ``str`` or ``None``

    :Example:

    >>> from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface
    >>> list_of_diseases = DiseaseOntInterface().pull()['name'].tolist()

    >>> abstract = '...a patient with pancreatitis, not Macrolipasemia' 
    >>> _disease_guess(problems=None, title=None, background=None,
    ...                abstract=abstract, image_caption=None, image_mention=None,
    ...                list_of_diseases=list_of_diseases)
    ...
    'pancreatitis'

    """
    # ToDO: this function is very computationally expensive and rather ineffective. Replace.
    generic = ('syndrome', 'disease')

    if isinstance(problems, str) and cln(problems).lower() in ('normal', 'none'):
        return 'normal'

    possible_diseases = list()
    for source in (image_caption, image_mention, problems, title, background, abstract):
        if isinstance(source, str) and len(source):
            source_clean = cln(BeautifulSoup(source, 'lxml').text.lower())
            for d in list_of_diseases:
                if d in source_clean:
                    possible_diseases.append([source_clean.find(d), d])

    no_substrings = _remove_substrings(possible_diseases)
    to_return = list(set([i[1] for i in no_substrings if i[1] not in generic]))

    if len(to_return):
        return "; ".join(sorted(to_return))
    # Try to fall back to `problems`.
    elif not len(to_return) and isinstance(problems, str):
        filtered_problems = [p for p in problems.split(";") if p in list_of_diseases]
        if len(filtered_problems):
            return "; ".join(sorted(filtered_problems))
        else:
            return None
    else:
        return None


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


def _illness_duration_guess(background, abstract, image_caption, image_mention):
    """

    Guess the duration of an illness. Unit: years.

    :param background: the background of the patient/study.
    :type background: ``str``
    :param abstract: a text abstract.
    :type abstract: ``str``
    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param image_mention: an element from the 'image_mention' column.
    :type image_mention: ``str``
    :return:
    :rtype: ``float`` or ``None``
    """
    for source in (background, abstract, image_caption, image_mention):
        if isinstance(source, str):
            duration_guess = _illness_duration_guess_engine(source.lower())
            if isinstance(duration_guess, float):
                return duration_guess
    else:
        return None


# ----------------------------------------------------------------------------------------------------------
# Imaging Technology (Modality)
# ----------------------------------------------------------------------------------------------------------


def _im_scan(source):
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
            # Scan `source` for subtypes (taking the opportunity to
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


def _im_drop_contradictions(values):
    """

    Check for and eliminate contradictions.

    :param values:
    :return:
    """
    for c in contradictions:
        if all(i in values for i in c):
            values = [m for m in values if m not in c]
    return values


def _im_formatter(dictionary):
    """

    Format the output; form: 'formal modality name: subtype1; subtype2; ...'.

    :param dictionary:
    :return:
    """
    modality = list(dictionary.keys())[0]
    modality_formal_name = terms_dict[modality][-1]
    values = _im_drop_contradictions(set(dictionary[modality]))
    if not len(values):
        return modality_formal_name
    else:
        return "{0}: {1}".format(modality_formal_name, "; ".join(sorted(values)))


def _imaging_modality_guess(abstract, image_caption, image_mention):
    """

    Guess the imaging technology (modality) used to take the picture.
    Preference: ``image_caption`` > ``image_mention`` > ``abstract``.

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
            scan_rslt = _im_scan(source=cln(source).lower())
            if isinstance(scan_rslt, dict) and len(scan_rslt.keys()) == 1:
                return _im_formatter(scan_rslt)
    else:
        return None  # capitulate


# ----------------------------------------------------------------------------------------------------------
# Image Plane
# ----------------------------------------------------------------------------------------------------------


def _image_plane_guess(image_caption):
    """

    Guess whether the plane of the image is 'axial', 'coronal' or 'sagittal'.

    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :return: see description.
    :rtype: ``str`` or ``None``
    """
    if not isinstance(image_caption, str):
        return None

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
    :return: enumerations present in `input_str`. Returns ``None`` if ``input_str`` is empty.
    :rtype: ``None`` or ``list``

    :Example:

    >>> _extract_enumerations('(1a) here we see... 2. whereas here we see...')
    ...
    ['1a', '2']
    """
    # Block floating point numbers, e.g., '...there are peaks at 2.3 and 3.28.'
    # See: http://stackoverflow.com/a/4703409/4898004. (Dropped second * to ignore '2.', but still catch (2.3).
    floating_point_numbers_removed = cln(re.sub(r"[-+]?\d*\.\d+", " ", input_str))

    # Further clean the input
    cleaned_input = cln(floating_point_numbers_removed, extent=2).replace("-", "").lower()

    if not len(cleaned_input):
        return None

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
    # Check for numeric enumeration, e.g., ['1', '2', '3', '4'].
    elif order_set(l) == list(map(str, range(1, len(l) + 1))):
        return True
    # Block simple enumerations that do not start at 1, e.g., ['2', '3', '4'].
    elif all(i.isdigit() for i in l):
        return False
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
    # Clean the input
    image_caption_reduced_confusion = multi_replace(image_caption, ['i.e.,', 'i.e.', 'e.g.,', 'e.g.', 'e.x.,', 'e.x.'])
    image_caption_clean_lower = cln(image_caption_reduced_confusion, extent=2).lower()

    # Check for markers of enumeration like '(a-e)', '(a,e)', '(b and c)'.
    match_on = ["[(|\[][a-z]" + i + "[a-z][)|\]]" for i in ("-", "and")] + ["[a-z]~[a-z]"]
    for regex in match_on:
        if len(re.findall(regex, image_caption_clean_lower)):
            return True
    else:
        # Fall back to standard enumerations
        enums = _extract_enumerations(input_str=image_caption_reduced_confusion)
        if isinstance(enums, list) and _enumerations_test(enums) and len(enums) >= enumerations_grid_threshold:
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
        # Example: '...which are indicated by the asterisks...'
        if any(i in image_caption_clean for i in match_on):
            features.add("{0}s".format(term))
        # Example: '...along the left side (arrows)...'
        elif len(re.findall(r'[(|\[]' + term + 's?[)|\]|,|;]', image_caption_clean)):
            features.add("{0}s".format(term))
        # Example: '...to the left (red arrows)...'
        elif len(re.findall(r'[(|\[].*? ' + term + 's?[)|\]|,|;]', image_caption_clean)):
            features.add("{0}s".format(term))

    # More markers may be added in the future
    if 'asterisks' not in features:
        if any("({0})".format(i) in image_caption_clean for i in ['*']):
            features.add("asterisks")

    return list(features)


# ----------------------------------------------------------------------------------------------------------
# Image Problems (Detected by Analysing their Associated Text)
# ----------------------------------------------------------------------------------------------------------


def _problematic_image_features(image_caption, image_caption_unique, enumerations_grid_threshold=2):
    """

    Currently determines whether or not the image contains
    arrows, asterisks or grids (multiple images arranged as as a grid of images).

    :param image_caption: an element from the 'image_caption' column.
    :type image_caption: ``str``
    :param image_caption_unique: whether or not the caption is unique with respect to the 'uid'.
    :type image_caption_unique: ``bool``
    :param enumerations_grid_threshold: the number of enumerations required to 'believe' the image is actually a
                                        'grid' of images, e.g., '1a. here we. 1b, rather' would suffice if this
                                        parameter is equal to `2`.
    :type enumerations_grid_threshold: ``int``
    :return: information on the presence of problematic image features.
    :rtype: ``tuple``

    :Example:

    >>> _problematic_image_features('1. left (green asterisk). 2. right(arrow; red and blue)...', image_caption_unique=True)
    ...
    ('arrows', 'asterisks', 'grids')
    """
    # Note: if ``image_caption_unique`` is not true, then the caption likely do not refer to the
    # current image, but with multiple images for the given 'uid'.
    if not image_caption_unique or not isinstance(image_caption, str) or not len(cln(image_caption, extent=2)):
        return None

    features = []

    # Look for markers
    features += _markers_guess(image_caption)

    if _enumerations_guess(image_caption, enumerations_grid_threshold):
        features.append('grids')
    
    return tuple(sorted(features)) if len(features) else None


# ----------------------------------------------------------------------------------------------------------
# Exposed Feature Extract Tool
# ----------------------------------------------------------------------------------------------------------


def _background_extract(d):
    """

    Extract background information for a parsed abstract.

    :param d: the ``d`` dictionary as evolved inside ``feature_extract()``.
    :type d: ``dict``
    :return: the study background/history (or case information, e.g., 'case_report').
    :rtype: ``None`` or ``str``
    """
    # Extract the background/history
    if isinstance(d['parsed_abstract'], dict):
        background = d['parsed_abstract'].get('background', None)
        if background is None:
            # Try to use 'case' information instead.
            case_keys = [k for k in d['parsed_abstract'] if 'case' in k]
            if len(case_keys) == 1:
                background = d['parsed_abstract'][case_keys[0]]
    else:
        background = None

    return background


def feature_extract(x, list_of_diseases, image_caption_unique):
    """

    Tool to extract text features from patient summaries.

    See the description of ``OpeniInterface().pull()`` for additional information.

    :param x: series passed though Pandas' ``DataFrame().apply()`` method, e.g.,
              ``df.apply(lambda x: feature_extract(x, list_of_diseases), axis=1)``.

              .. note::

                   The dataframe must contain 'title', 'abstract', 'image_caption', 'image_mention'
                   and 'journal_title' columns.

    :type x: ``Pandas Series``
    :param list_of_diseases: a list of diseases (e.g., via ``DiseaseOntInterface().pull()['name'].tolist()``)
    :type list_of_diseases: ``list``
    :param image_caption_unique: whether or not the caption is unique with respect to the 'uid'.
    :type image_caption_unique: ``bool``
    :return: dictionary with the keys listed in the description.
    :rtype: ``dict``
    """
    # Note: it may be useful to use 'unescape' here, e.g., unescape(x[c])...but no reason to support doing so currently.
    # However, if this were to occur, 'abstract' would need to be excluded in the service of ``_abstract_parser()``.
    cols_to_unpack = ('abstract', 'problems', 'title', 'image_caption', 'image_mention', 'article_type')
    abstract, problems, title, image_caption, image_mention, article_type = [x[c] for c in cols_to_unpack]

    d = {'parsed_abstract': _abstract_parser(abstract)}
    background = _background_extract(d)

    # Extract diagnosis -- will typically only work for MedPix images
    d['diagnosis'] = d['parsed_abstract'].get('diagnosis', None) if isinstance(d['parsed_abstract'], dict) else None

    # Fall back
    if d['diagnosis'] is None:
        d['diagnosis'] = _disease_guess(problems=problems, title=title, background=background,
                                        # Block use of 'abstract' when the article is not clinical.
                                        abstract=abstract if article_type in CLINICAL_ARTICLE_TYPES else None,
                                        image_caption=image_caption if image_caption_unique else None,
                                        image_mention=image_mention,
                                        list_of_diseases=list_of_diseases)

    if isinstance(d['diagnosis'], str):
        d['diagnosis'] = d['diagnosis'].lower()

    pairs = [('age', _patient_age_guess), ('sex', _patient_sex_guess), ('illness_duration_years', _illness_duration_guess)]
    for (k, func) in pairs:
        d[k] = func(background, abstract, image_caption, image_mention)

    d['imaging_modality_from_text'] = _imaging_modality_guess(abstract, image_caption, image_mention)
    d['image_plane'] = _image_plane_guess(image_caption)
    d['image_problems_from_text'] = _problematic_image_features(image_caption, image_caption_unique)
    d['ethnicity'], eth_sex = _ethnicity_guess(background, abstract, image_caption, image_mention)

    if d['sex'] is None and eth_sex is not None:
        d['sex'] = eth_sex

    return d
