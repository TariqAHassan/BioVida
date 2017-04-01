# coding: utf-8

"""

    Dictionaries of Open-i Params
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from biovida.support_tools.support_tools import cln
from biovida.support_tools.support_tools import dict_reverse


def _parser(s):
    """
    
    Tool to convert the lists on https://openi.nlm.nih.gov/services.php?it=xg to dicts.
    
    :param s: a string copied from the URL above
    :type s: ``str``
    :return: a dictionary of params.
    :rtype: ``dict``
    """
    # Split the string into keys and values
    split_string = [i.split('(') for i in cln(s).split(')')]

    # Define a lambda to clean the values
    def lower_join(x):
        return '_'.join(cln(x).split()).strip().lower()

    # Generate the dict
    return {cln(k[1]): lower_join(k[0]) for k in split_string if set(k) != {''}}


openi_video_params = {
    '1': 'true'
}


openi_image_type_params = {
    'c': 'ct',
    'g': 'graphic',
    'm': 'mri',
    'mc': 'microscopy',
    'p': 'pet',
    'ph': 'photograph',
    'u': 'ultrasound',
    'x': 'x_ray',
    'xg': 'exclude_graphics',
    'xm': 'exclude_multipanel'
}


openi_image_type_modality_full = {
    # Written to cohere with yield of `cancer_imaging_parameters.py`'s
    # `CancerImageArchiveParams().dicom_modality_abbreviations()` method.
    'c': 'Computed Tomography (CT)',
    'g': 'Graphic',
    'm': 'Magnetic Resonance Imaging (MRI)',
    'mc': 'Microscopy',
    'p': 'Positron Emission Tomography (PET)',
    'ph': 'Photograph',
    'u': 'Ultrasound',
    'x': 'X-Ray',
}


openi_rankby_params = {
    'r': 'newest',
    'o': 'oldest',
    'd': 'diagnosis',
    'e': 'etiology',
    'oc': 'outcome',
    'pr': 'prevention',
    'pg': 'prognosis',
    't': 'treatment'
}


openi_subset_params = {
    'b': 'basic_science',
    'c': 'clinical_journals',
    'e': 'ethics',
    's': 'systematic_reviews',
    'x': 'chest_x_rays'
}


openi_collection_params = {
    'pmc': 'pubmed',
    'hmd': 'history_of_medicine',
    'iu': 'indiana_u_xray',
    'mpx': 'medpix',
    'usc': 'usc_anatomy',
}


openi_fields_params = {
    'a': 'authors',
    'ab': 'abstracts',
    'c': 'captions',
    'm': 'mentions',
    'msh': 'mesh',
    't': 'titles'
}


openi_article_type_params = {
    'ab': 'abstract',
    'bk': 'book_review',
    'bf': 'brief_report',
    'cr': 'case_report',
    'dp': 'data_paper',
    'di': 'discussion',
    'ed': 'editorial',
    'ib': 'in_brief',
    'in': 'introduction',
    'lt': 'letter',
    'mr': 'meeting_report',
    'ma': 'method_article',
    'ne': 'news',
    'ob': 'obituary',
    'pr': 'product_review',
    'or': 'oration',
    're': 'reply',
    'ra': 'research_article',
    'rw': 'review_article',
    'sr': 'systematic_reviews',
    'rr': 'radiology_report',
    'os': 'orthopedic_slides',
    'hs': 'historical_slide',
    'ot': 'other'
}


openi_specialties_params = {
    'b': 'behavioral_sciences',
    'bc': 'biochemistry',
    'c': 'cancer',
    'ca': 'cardiology',
    'cc': 'critical_care',
    'd': 'dentistry',
    'de': 'dermatology',
    'dt': 'drug_therapy',
    'e': 'emergency_medicine',
    'eh': 'environmental_health',
    'en': 'endocrinology',
    'f': 'family_practice',
    'g': 'gastroenterology',
    'ge': 'genetics',
    'gr': 'geriatrics',
    'gy': 'gynecology_&_obstetrics',
    'h': 'hematology',
    'i': 'immunology',
    'id': 'infectious_diseases',
    'im': 'internal_medicine',
    'n': 'nephrology',
    'ne': 'neurology',
    'nu': 'nursing',
    'o': 'ophthalmology',
    'or': 'orthopedics',
    'ot': 'otolaryngology',
    'p': 'pediatrics',
    'pu': 'pulmonary_diseases',
    'py': 'psychiatry',
    'r': 'rheumatology',
    's': 'surgery',
    't': 'toxicology',
    'u': 'urology',
    'v': 'vascular_diseases',
    'vi': 'virology'
}


openi_api_search_params = {
    'video': ('&vid', openi_video_params),
    'image_type': ('&it', openi_image_type_params),
    'rankby': ('&favor', openi_rankby_params),
    'article_type': ('&at', openi_article_type_params),
    'subset': ('&sub', openi_subset_params),
    'collection': ('&coll', openi_collection_params),
    'fields': ('&fields', openi_fields_params),
    'specialties': ('&sp', openi_specialties_params),
}


def openi_search_information():
    """

    Returns a dictionary of the form:
        ``{search_category: (URL_Parameter, {search term: URL_Parameter}), ...}``

    :return: search information for the Open-i API.
    :rtype: ``dict``
    """
    # Params in the order the Open-i API expects
    ordered_params = ['query', '&it', '&favor', '&at', '&vid', '&sub', '&coll', '&fields', '&sp']

    # Return the openi_api_search_params dict with the dicts nested therein reversed.
    return {k: (v[0], dict_reverse(v[1])) for k, v in openi_api_search_params.items()}, ordered_params
