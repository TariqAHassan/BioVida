"""

    Unify BioVida APIs
    ~~~~~~~~~~~~~~~~~~

"""
# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint
from collections import defaultdict

# Import Interfaces
from biovida.images.openi_interface import OpeniInterface
from biovida.diagnostics.hsdn_interface import HsdnInterface
from biovida.genomics.disgenet_interface import DisgenetInterface
from biovida.images.cancer_image_interface import CancerImageInterface
from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface

# Support Tools
from biovida.support_tools.support_tools import is_int
from biovida.support_tools.support_tools import items_null

# Temporary
from biovida.support_tools.printing import pandas_pprint
from biovida.images.image_processing import ImageProcessing

# Start tqdm
tqdm.pandas(desc='status')


# ----------------------------------------------------------------------------------------------------------
# Interface Integration
# ----------------------------------------------------------------------------------------------------------


class _ImagesInterfaceIntegration(object):
    """

    Tools to Unify BioVida Image Interfaces

    """

    def _open_i_prep(self, cache_record_db):
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
        openi_subsection['source'] = ['openi'] * openi_subsection.shape[0]

        # Apply rename and return
        return openi_subsection.rename(columns=openi_col_rename)

    def _cancer_img_prep(self, cache_record_db):
        """

        :param cache_record_db:
        :return:
        """
        # Define columns to keep
        cancer_img_columns = ['series_instance_uid', 'series_description', 'modality_full', 'age', 'sex',
                              'cancer_type', 'query', 'query_time', 'conversion_success', 'converted_files_paths']

        # Column name changes (based on `_open_i_prep`).
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
        cancer_img_subsection['source'] = ['tcia'] * cancer_img_subsection.shape[0]

        # Apply rename and return
        return cancer_img_subsection.rename(columns=cancer_img_col_rename)

    def prep_class_dict_gen(self):
        """

        Generate a dictionary which maps image interface classes to
        the methods designed to handle them.

        :return: a dictionary mapping class names to functions.
        :rtype: ``dict``
        """
        return {'OpeniInterface': self._open_i_prep, 'CancerImageInterface': self._cancer_img_prep}

    def integration(self, interfaces):
        """

        Note: classes are assumed to have a class attr called ``cache_record_db``.

        :param interfaces:
        :rtype interfaces: ``tuple``, ``list``, ``OpeniInterface`` class or ``CancerImageInterface`` class.
        :return:
        """
        prep_class_dict = self.prep_class_dict_gen()

        # Handle instances being passed 'raw'
        interfaces = [interfaces] if not isinstance(interfaces, (list, tuple)) else interfaces

        frames = list()
        for class_instance in interfaces:
            func = prep_class_dict[type(class_instance).__name__]
            database = getattr(class_instance, 'cache_record_db')
            if 'DataFrame' not in str(type(database)):
                raise ValueError("The {0} instance's '{1}' database cannot be `None`.".format(
                    type(class_instance).__name__, 'cache_record_db'))
            frames.append(func(database))

        return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------
# Disease Ontology Integration
# ----------------------------------------------------------------------------------------------------------


class _DiseaseOntologyIntegration(object):
    """

    Integration of Disease Ontology information.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
    Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
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
          Form ``{'synonym': ['name', 'name', ...], ...}``

        :param ontology_df: yeild of ``DiseaseOntInterface().pull()``
        :type ontology_df: ``Pandas DataFrame``
        :return: see method description.
        :rtype: ``tuple``
        """
        # Init
        ont_name_dict, ont_synonym_dict = dict(), dict()
        ont_name_dict_nest_keys = ('disease_family', 'synonym', 'diagnosis_definition')

        def str_split(s, split_on='; '):
            return tuple(s.split('; ')) if isinstance(s, str) else s

        for name, is_a, synonym, defn in zip(*[ontology_df[c] for c in ('name', 'is_a', 'synonym', 'def')]):
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

        return ont_name_dict, ont_name_dict_nest_keys, ont_synonym_dict

    def __init__(self, cache_path=None, verbose=True):
        # Load the databse
        ontology_df = DiseaseOntInterface(cache_path=cache_path, verbose=verbose).pull()

        # Obtain dictionaries
        self.ont_name_dict, self.ont_name_dict_nest_keys, self.ont_synonym_dict = self._dis_ont_dict_gen(ontology_df)

        # Conver `ont_name_dict_nest_keys` to an empty dict.
        self.empty_nest_dict = dict.fromkeys(self.ont_name_dict_nest_keys, np.NaN)

        # Extract keys from the two dictionaries passed
        self.ont_name_dict_keys = tuple(self.ont_name_dict.keys())
        self.ont_synonym_dict_keys = tuple(self.ont_synonym_dict.keys())

    def _synonym_match(self, synonym):
        """

        Maps a disease synonym to an actual disease name.
        The 'synonym' key of the dictionary which is returned will have `synonym` removed
        and will have a disease names which are mapped to `synonym` installed in its place.

        :param synonym: a disease synonym.
        :param synonym: ``str``
        :return:
        :rtype: ``dict``
        """
        # Mapping from synonym to related diseases
        ont_dis_names = self.ont_synonym_dict[synonym]

        # Simply use the first disease name related to the synonym.
        # Note: this *assumes* that which 'name' is chosen from the list is irrelevant.
        # If the disease ontology database is not consistant, this assumption is invalid.
        disease_info = deepcopy(self.ont_name_dict[ont_dis_names[0]])
        # Remove the synonym from the 'synonym' key and add 'ont_dis_names'
        if isinstance(disease_info['synonym'], tuple):
            synonym_new = [i for i in disease_info['synonym'] if i != synonym] + ont_dis_names
        else:
            synonym_new = [ont_dis_names]

        # Add to `disease_info` (and remove any possible duplicates)
        disease_info['synonym'] = tuple(set(synonym_new))

        return disease_info

    def _find_disease_info_raw(self, disease):
        """

        Try to match the input (`disease`) to information in the Disease Ontology Database.

        :param disease: a disease name.
        :param disease: ``str``
        :return:
        :rtype: ``dict`` or ``None``
        """
        if items_null(disease) or disease is None:
            return None
        elif disease in self.ont_name_dict:
            return self.ont_name_dict[disease]
        elif disease in self.ont_synonym_dict:
            return self._synonym_match(synonym=disease)
        else:
            return None

    def _find_disease_info(self, disease, fuzzy_threshold):
        """

        :param disease: a disease name.
        :param disease: ``str``
        :param fuzzy_threshold: an intiger on [0, 100].
        :type fuzzy_threshold: ``int``, `bool`, ``None``
        :return:
        """
        # ToDo: add memorizing of fuzzy matches
        # Try matching the string raw (i.e., 'as is').
        raw_rslt = self._find_disease_info_raw(disease)
        if isinstance(raw_rslt, dict):
            return raw_rslt

        # Eject if fuzzy matching is disabled
        if not is_int(fuzzy_threshold):
            return self.empty_nest_dict

        # Try using `ont_name_dict`
        name_fuzzy_match, threshold = process.extractOne(disease, self.ont_name_dict_keys)
        if threshold >= fuzzy_threshold:
            return self.ont_name_dict[name_fuzzy_match]

        # Try using `ont_synonym_dict`
        synonym_fuzzy_match, threshold = process.extractOne(disease, self.ont_synonym_dict_keys)
        if threshold >= fuzzy_threshold:
            return self._synonym_match(synonym_fuzzy_match)
        else:
            return self.empty_nest_dict

    def integration(self, combined_df, fuzzy_threshold=False):
        """

        :param combined_df:
        :type combined_df: ``Pandas DataFrame``
        :param fuzzy_threshold:
        :return:
        """
        if fuzzy_threshold is True:
            raise ValueError("`fuzzy_threshold` cannot be `True`. Please provide a specific intiger on (0, 100].")

        # Extract diagnosis information using the Disease Ontology database
        if is_int(fuzzy_threshold):
            # Use progress map to show progress
            disease_ontology_data = combined_df['diagnosis'].progress_map(
                lambda x: self._find_disease_info(x, fuzzy_threshold))
        else:
            # When `fuzzy_threshold` if off, this method is sufficently fast that a progress bar is not needed
            disease_ontology_data = combined_df['diagnosis'].map(lambda x: self._find_disease_info(x, fuzzy_threshold))

        # Convert `disease_ontology_data` to a dataframe
        disease_ontology_addition = pd.DataFrame(disease_ontology_data.tolist())

        # Add the columns in `disease_ontology_addition` to `combined_df`.
        for c in self.ont_name_dict_nest_keys:
            combined_df[c] = disease_ontology_addition[c]

        return combined_df


# ----------------------------------------------------------------------------------------------------------
# Tools to Add Data by Matching Against Disease and Disease Synonyms.
# ----------------------------------------------------------------------------------------------------------


def _disease_synonym_match_battery(disease, synonyms, resource_dict, fuzzy_threshold):
    """

    Try to match ``disease`` and ``synonyms`` in ``resource_dict``

    :param disease:
    :param synonyms:
    :param resource_dict:
    :param fuzzy_threshold: must be an intiger for fuzzy matching to be enabled.
    :return:
    """
    # Extract the keys
    lookup_dict_keys = tuple(resource_dict.keys())

    # Try disease 'as is'
    if disease in resource_dict:
        return resource_dict[disease]

    # Search through synonyms
    if isinstance(synonyms, tuple):
        for s in synonyms:
            if s in resource_dict:
                return resource_dict[s]

    # Eject if fuzzy matching is disabled
    if not is_int(fuzzy_threshold):
        return np.NaN

    # Try Fuzzy matching on `disease`
    disease_fuzzy_match, threshold = process.extractOne(disease, lookup_dict_keys)
    if threshold >= fuzzy_threshold:
        return resource_dict[disease_fuzzy_match]

    # Try Fuzzy matching on `synonyms`
    if not isinstance(synonyms, tuple):
        return np.NaN
    else:
        for s in synonyms:
            synonym_fuzzy_match, threshold = process.extractOne(s, lookup_dict_keys)
            if threshold >= fuzzy_threshold:
                return resource_dict[synonym_fuzzy_match]
        else:
            return np.NaN  # capitulate


def _resource_integration(data_frame, resource_dict, fuzzy_threshold, new_column_name):
    """
    
    :param data_frame: 
    :type data_frame: ``Pandas DataFrame``
    :param resource_dict:
    :type resource_dict: ``type``
    :param fuzzy_threshold:
    :type fuzzy_threshold: ``int``, ``False`` or ``None``
    :param new_column_name: 
    :type new_column_name: ``str``
    :return: 
    """
    if fuzzy_threshold is True:
        raise ValueError("`fuzzy_threshold` cannot be `True`. Please specify a specific intiger on [0, 100].")

    missing_column_error_message = "`data_frame` must contain a '{0}' column." \
                                   "Call ``_DiseaseOntologyIntegration().disease_ont_integration()``"

    if 'diagnosis' not in data_frame.columns:
        raise AttributeError(missing_column_error_message.format('diagnosis'))
    elif 'synonym' not in data_frame.columns:
        raise AttributeError(missing_column_error_message.format('synonym'))

    # Map gene-disease information onto the dataframe
    if is_int(fuzzy_threshold):
        # Use progress bar (fuzzy matching is expensive)
        rslt = data_frame.progress_apply(
            lambda x: _disease_synonym_match_battery(disease=x['diagnosis'],
                                                     synonyms=x['synonym'],
                                                     resource_dict=resource_dict,
                                                     fuzzy_threshold=fuzzy_threshold), axis=1)
    else:
        # No need for a progress bar.
        rslt = data_frame.apply(
            lambda x: _disease_synonym_match_battery(disease=x['diagnosis'],
                                                     synonyms=x['synonym'],
                                                     resource_dict=resource_dict,
                                                     fuzzy_threshold=fuzzy_threshold), axis=1)

    # Add the `rslt` series to `data_frame`
    data_frame[new_column_name] = rslt

    return data_frame


# ----------------------------------------------------------------------------------------------------------
# HSDN Interface (Symptomatology)
# ----------------------------------------------------------------------------------------------------------


class _HsdnIntegration(object):
    """

    Integration of Human Symptoms Disease Network information.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
    Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    """

    def _disease_symptom_dict_gen(self, hsdn_db):
        """

        Tool to create a dictionary mapping disease to symptoms.

        :param hsdn_db:
        :type hsdn_db: ``Pandas DataFrame``
        :return: a dictionary of the form ``{disease name: [symptom, symptom, symptom, ...], ...}``
        :rtype: ``dict``
        """
        d = defaultdict(list)
        for disease, symptom in zip(hsdn_db['common_disease_name'], hsdn_db['common_symptom_term']):
            d[disease.lower()] += [symptom.lower()]
        return {k: tuple(v) for k, v in d.items()}

    def __init__(self, cache_path=None, verbose=True):
        # Load the HSDN database
        hsdn_db = HsdnInterface(cache_path=cache_path, verbose=verbose).pull()

        # Create a disease-symptoms mapping
        self.disease_symptom_dict = self._disease_symptom_dict_gen(hsdn_db)

    def integration(self, data_frame, fuzzy_threshold=False):
        """

        :param data_frame:
        :param fuzzy_threshold:
        :return:
        """
        return _resource_integration(data_frame=data_frame,
                                     resource_dict=self.disease_symptom_dict,
                                     fuzzy_threshold=fuzzy_threshold,
                                     new_column_name='known_associated_symptoms')


# ----------------------------------------------------------------------------------------------------------
# Disgenet Integration
# ----------------------------------------------------------------------------------------------------------


class _DisgenetIntegration(object):
    """

    Integration of DisGeNET information.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
    Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    """

    def _disease_gene_dict_gen(self, disgenet_df):
        """

        :param disgenet_df: yeild of ``DisgenetInterface().pull('all)``.
        :type disgenet_df: ``Pandas DataFrame``
        :return: dictionary of the form ``{'disease_name': [('gene_name', disgenet score), ...], ...}``.
        :rtype: ``dict``
        """
        d = defaultdict(list)
        cols = zip(*[disgenet_df[c] for c in ('disease_name', 'gene_name', 'score')])
        for disease_name, gene_name, score in cols:
            d[disease_name].append((gene_name, score))
        return {k: tuple(sorted(v, key=lambda x: x[0])) for k, v in d.items()}

    def __init__(self, cache_path=None, verbose=True):
        # Load the database
        disgenet_df = DisgenetInterface(cache_path=cache_path, verbose=verbose).pull('all')

        # Extract the relevant information in `disgenet_df` to a dictionary.
        self.disease_gene_dict = self._disease_gene_dict_gen(disgenet_df)

    def integration(self, data_frame, fuzzy_threshold=False):
        """

        :param data_frame:
        :param fuzzy_threshold:
        :return:
        """
        return _resource_integration(data_frame=data_frame,
                                     resource_dict=self.disease_gene_dict,
                                     fuzzy_threshold=fuzzy_threshold,
                                     new_column_name='known_associated_genes')


# ----------------------------------------------------------------------------------------------------------
# Unify
# ----------------------------------------------------------------------------------------------------------


def _try_fuzzywuzzy_import():
    """

    Try to import the ``fuzzywuzzy`` library.

    :return:
    """
    try:
        from fuzzywuzzy import process
        global process
    except ImportError:
        error_msg = "`fuzzy_threshold` requires the `fuzzywuzzy` library,\n" \
                    "which can be installed with `$ pip install fuzzywuzzy`.\n" \
                    "For best performance, it is also recommended that python-Levenshtein is installed.\n" \
                    "(`pip install python-levenshtein`)."
        raise error_msg


def unify(interfaces, cache_path=None, verbose=True, fuzzy_threshold=False):
    """

    Tool to Unify Image Interfaces (currently ``OpeniInterface`` and ``CancerImageInterface``)
    with Diagnostic and Genomic Data

    :param interfaces:
    :param cache_path:
    :param verbose:
    :param fuzzy_threshold:
    :return:
    """
    if is_int(fuzzy_threshold):
        _try_fuzzywuzzy_import()

    # Combine Instances
    combined_df = _ImagesInterfaceIntegration().integration(interfaces=interfaces)

    # Disease Ontology
    combined_df = _DiseaseOntologyIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    # Human Symptoms Disease Network
    combined_df = _HsdnIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    # Disgenet
    combined_df = _DisgenetIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    return combined_df


















