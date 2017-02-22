"""

    Unify BioVida APIs
    ~~~~~~~~~~~~~~~~~~

"""
import numpy as np
import pandas as pd
from copy import deepcopy
from pprint import pprint

from fuzzywuzzy import process  # ToDo: make optional

from biovida.images.openi_interface import OpeniInterface
from biovida.genomics.disgenet_interface import DisgenetInterface
from biovida.images.cancer_image_interface import CancerImageInterface
from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface

from biovida.support_tools.support_tools import items_null

# Temp
from biovida.images.image_processing import ImageProcessing
from biovida.support_tools.printing import pandas_pprint


# from biovida.images._resources.cancer_image_parameters import CancerImgArchiveParams
# CancerImgArchiveParams().dicom_modality_abbreviations()

# Open-i
opi = OpeniInterface()
# opi.search(query=None, image_type=['mri', 'ct'])
# search_df = opi.pull(download_limit=2000)

# TCIA
cii = CancerImageInterface(API_KEY)
# cii.search(location='extremities', modality='mri')
# cdf = cii.pull(patient_limit=1, allowed_modalities='mri')
# pandas_pprint(cii.cache_record_db, full_cols=True)


# ----------------------------------------------------------------------------------------------------------
# Interface Integration
# ----------------------------------------------------------------------------------------------------------


def open_i_prep(cache_record_db):
    """

    :param cache_record_db:
    :return:
    """
    # Deep copy the input to prevent mutating the original in memory.
    cache_record_db_cln = cache_record_db.copy(deep=True)

    # Column which provides a guess, based on the text, on which imaging modality created the image.
    cache_record_db_cln['modality_best_guess'] = cache_record_db_cln.apply(
        lambda x: x['caption_imaging_modality'] if not items_null(x['caption_imaging_modality']) else x['modality_full'],
        axis=1
    )

    # Convert the 'converted_files_path' column from a series of string to a series of tuples.
    cache_record_db_cln['converted_files_path'] = cache_record_db_cln['converted_files_path'].map(
        lambda x: tuple([x]) if not isinstance(x, tuple) else x, na_action='ignore'
    )

    # Define columns to keep
    openi_columns = ['image_id', 'image_caption', 'modality_best_guess', 'age', 'sex',
                     'diagnosis', 'query', 'query_time', 'download_success', 'converted_files_path']

    # Column name changes
    openi_col_rename = {'converted_files_path': 'files_path', 'download_success': 'harvest_success'}

    # Define subsection based on `openi_columns`
    openi_subsection = cache_record_db_cln[openi_columns]

    # Add a column to allow the user to identify the API which provided the data
    openi_subsection['source_api'] = ['openi'] * openi_subsection.shape[0]

    # Apply rename and return
    return openi_subsection.rename(columns=openi_col_rename)


def cancer_img_prep(cache_record_db):
    """

    :param cache_record_db:
    :return:
    """
    # Define columns to keep
    cancer_img_columns = ['series_instance_uid', 'series_description', 'modality_full', 'age', 'sex',
                          'cancer_type', 'query', 'query_time', 'conversion_success', 'converted_files_paths']

    # Column name changes (based on `open_i_prep`).
    cancer_img_col_rename = {'series_instance_uid': 'image_id',
                             'series_description': 'image_caption',
                             'modality_full': 'modality_best_guess',
                             'cancer_type': 'diagnosis',
                             'conversion_success': 'harvest_success',
                             'converted_files_paths': 'files_path'}

    # Deep copy the input to prevent mutating the original in memory.
    cache_record_db_cln = cache_record_db.copy(deep=True)

    # Define subsection based on `cancer_img_columns`
    cancer_img_subsection = cache_record_db_cln[cancer_img_columns]

    # Add a column to allow the user to identify the API which provided the data
    cancer_img_subsection['source_api'] = ['tcia'] * cancer_img_subsection.shape[0]

    # Apply rename and return
    return cancer_img_subsection.rename(columns=cancer_img_col_rename)


def refine_and_combine(interfaces):
    """

    Note: classes are assumed to have a class attr called ``cache_record_db``.

    :param interfaces:
    :rtype interfaces: ``tuple``, ``list``, ``OpeniInterface`` class or ``CancerImageInterface`` class.
    :return:
    """
    _prep_class_dict = {
        'OpeniInterface': open_i_prep,
        'CancerImageInterface': cancer_img_prep
    }

    # Handle instances being passed 'raw'
    interfaces = [interfaces] if not isinstance(interfaces, (list, tuple)) else interfaces

    frames = list()
    for class_instance in interfaces:
        func = _prep_class_dict[type(class_instance).__name__]
        database = getattr(class_instance, 'cache_record_db')
        if 'DataFrame' not in str(type(database)):
            raise ValueError("The {0} instance's '{1}' database cannot be None.".format(
                type(class_instance).__name__, 'cache_record_db')
            )
        frames.append(func(database))

    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------
# Disease Ontology Integration
# ----------------------------------------------------------------------------------------------------------


class _DiseaseOntologyIntegration(object):
    """

    :param ontology_df: yeild of ``DiseaseOntInterface().pull()``.
    :type ontology_df: ``Pandas DataFrame``
    """

    def _dis_ont_dict_gen(self, ontology_df):
        """

        Conver the information obtained from ``DiseaseOntInterface().pull()`` into:

        - a nested dictionary with ``ontology_df``'s 'name' column as the outer key (``ont_name_dict``).
          Form: ``{'name': {'disease_family' ('is_a'): tuple or None,
                            'synonym': tuple or None,
                            'diagnosis_definition' ('def'): str or None},
                  ...}``

        - the keys of the nested dictionaries in ``ont_name_dict``

        - a dictionary with 'synonym' as keys and related names in a list:
          Form ``{'synonym': ['name', 'name',...], ...}``

        :param ontology_df: yeild of ``DiseaseOntInterface().pull()``
        :type ontology_df:
        :return: see method description.
        :rtype: ``tuple``
        """
        # Init
        ont_name_dict, ont_synonym_dict = dict(), dict()
        ont_name_dict_keys = ('disease_family', 'synonym', 'diagnosis_definition')

        def str_split(s, split_on='; '):
            return tuple(s.split('; ')) if isinstance(s, str) else s

        cols = zip(*[ontology_df[c] for c in ('name', 'is_a', 'synonym', 'def')])
        for name, is_a, synonym, defn in cols:
            synonym_split = str_split(synonym)
            if not items_null(name):
                # Update `ont_name_dict`
                ont_name_dict[name] = {'disease_family': str_split(is_a),
                                       'synonym': synonym_split,
                                       'diagnosis_definition': defn}

                # Update `ont_synonym_dict`
                if isinstance(synonym_split, tuple):
                    for s in synonym_split:
                        if s not in ont_synonym_dict:
                            ont_synonym_dict[s] = [name]
                        # Check a duplicate is not added under a given synonym
                        elif name not in ont_synonym_dict[s]:
                            ont_synonym_dict[s] += [name]

        return ont_name_dict, ont_name_dict_keys, ont_synonym_dict

    def __init__(self, ontology_df):
        # Obtain dictionaries
        self.ont_name_dict, ont_name_dict_keys, self.ont_synonym_dict = self._dis_ont_dict_gen(ontology_df)

        # Conver `ont_name_dict_keys` to an empty dict.
        self.empty_nest_dict = dict.fromkeys(ont_name_dict_keys, np.NaN)

        # Extract keys from the two dictionaries passed
        self.ont_name_dict_keys = set(self.ont_name_dict.keys())
        self.ont_synonym_dict_keys = set(self.ont_synonym_dict.keys())

    def _synonym_match(self, disease):
        """

        :param disease:
        :return:
        """
        # Simply use the first name related to the synonym.
        # Note: this *assumes* that which 'name' is chosen from the list is irrelevant.
        ont_dis_name = self.ont_synonym_dict[disease][0]
        disease_info = deepcopy(self.ont_name_dict[ont_dis_name])
        # Remove the synonym from the 'synonym' key and add 'ont_dis_name' in its place.
        if isinstance(disease_info['synonym'], tuple):
            disease_info['synonym'] = tuple([i for i in disease_info['synonym'] if i != disease] + [ont_dis_name])
        else:
            disease_info['synonym'] = tuple([ont_dis_name])
        return disease_info

    def _find_disease_info_raw(self, disease):
        """

        Try to match the input (`disease`) to information in the Disease Ontology Database.

        :param disease:
        :return:
        """
        if items_null(disease) or disease is None:
            return None
        elif disease in ont_name_dict:
            return self.ont_name_dict[disease]
        elif disease in ont_synonym_dict:
            return self._synonym_match(disease)
        else:
            return None

    def _find_disease_info(self, disease, fuzzy_threshold):
        """

        :param disease:
        :return:
        """
        # Try matching the string raw (i.e., 'as is').
        raw_rslt = _find_disease_info_raw(disease)
        if isinstance(raw_rslt, dict):
            return raw_rslt

        # Eject if fuzzy matching is disabled.
        if not isinstance(fuzzy_threshold, int):
            return self.empty_nest_dict

        # Try using `ont_name_dict`
        names_fuzzy_match, threshold = process.extractOne(disease, self.ont_name_dict_keys)
        if threshold >= fuzzy_threshold:
            return self.ont_name_dict[names_fuzzy_match]

        # Try using `ont_synonym_dict`
        synonyms_fuzzy_match, threshold = process.extractOne(disease, self.ont_synonym_dict_keys)
        if threshold >= fuzzy_threshold:
            return self._synonym_match(synonyms_fuzzy_match)
        else:
            return self.empty_nest_dict

    def disease_ont_integration(self, combined_df, fuzzy_threshold=90):
        """

        :param combined_df:
        :param fuzzy_threshold:
        :return:
        """
        # Extract diagnosis information using the Disease Ontology database
        disease_ontology_data = combined_df['diagnosis'].map(lambda x: self._find_disease_info(x, fuzzy_threshold))

        # Convert to a dataframe
        disease_ontology_addition = pd.DataFrame(disease_ontology_data.tolist())

        # Add the information to `combined_df`.
        for c in ('synonym', 'disease_family', 'diagnosis_definition'):
            combined_df[c] = disease_ontology_addition[c]

        return combined_df


combined_df = refine_and_combine(interfaces=[opi, cii])

# Disease Ontology
doi = DiseaseOntInterface()
ontology_df = doi.pull()

# Disgenet
# dna = DisgenetInterface()
# gdf = dna.pull('all')















