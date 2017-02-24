"""

    Unify BioVida APIs
    ~~~~~~~~~~~~~~~~~~

"""
# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

# Import Interfaces
from biovida.genomics.disgenet_interface import DisgenetInterface
from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface
from biovida.diagnostics.disease_symptoms_interface import DiseaseSymptomsInterface

# Support Tools
from biovida.support_tools.support_tools import is_int
from biovida.support_tools.support_tools import header
from biovida.support_tools.support_tools import items_null

# Temporary
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

        A tool to clean and standardize  an ``OpeniInterface`` instance's cache record database

        :param cache_record_db: the cache record database from an ``OpeniInterface`` instance.
        :type cache_record_db: ``Pandas DataFrame``
        :return: a cleaned and standardize ``cache_record_db`` 
        :rtype: ``Pandas DataFrame``
        """
        # Deep copy the input to prevent mutating the original in memory.
        cache_record_db_cln = cache_record_db.copy(deep=True)

        # Column which provides a guess, based on the text, on which imaging modality created the image.
        cache_record_db_cln['modality_best_guess'] = cache_record_db_cln.apply(
            lambda x: x['caption_imaging_modality'] if not items_null(x['caption_imaging_modality']) else x['modality_full'],
            axis=1
        )

        # Convert the 'cached_images_path' column from a series of string to a series of tuples.
        cache_record_db_cln['cached_images_path'] = cache_record_db_cln['cached_images_path'].map(
            lambda x: tuple([x]) if not isinstance(x, tuple) else x, na_action='ignore'
        )

        # Define columns to keep
        openi_columns = ['image_id', 'image_caption', 'modality_best_guess', 'age', 'sex',
                         'diagnosis', 'query', 'query_time', 'download_success', 'cached_images_path']

        # Column name changes
        openi_col_rename = {'diagnosis': 'disease',
                            'cached_images_path': 'files_path',
                            'download_success': 'harvest_success'}

        # Define subsection based on `openi_columns`
        openi_subsection = cache_record_db_cln[openi_columns]

        # Add a column to allow the user to identify the API which provided the data
        openi_subsection['source_api'] = ['openi'] * openi_subsection.shape[0]

        # Apply rename and return
        return openi_subsection.rename(columns=openi_col_rename)

    def _cancer_img_prep(self, cache_record_db):
        """

        A tool to clean and standardize  an ``CancerImageInterface`` instance's cache record database

        :param cache_record_db: the cache record database from an ``CancerImageInterface`` instance.
        :type cache_record_db: ``Pandas DataFrame``
        :return: a cleaned and standardize ``cache_record_db`` 
        :rtype: ``Pandas DataFrame``
        """
        # Define columns to keep
        cancer_img_columns = ['series_instance_uid', 'series_description', 'modality_full', 'age', 'sex',
                              'cancer_type', 'query', 'query_time', 'conversion_success', 'cached_images_path']

        # Column name changes (based on `_open_i_prep`).
        cancer_img_col_rename = {'series_instance_uid': 'image_id',
                                 'series_description': 'image_caption',
                                 'modality_full': 'modality_best_guess',
                                 'cancer_type': 'disease',
                                 'conversion_success': 'harvest_success',
                                 'cached_images_path': 'files_path'}

        # Deep copy the input to prevent mutating the original in memory.
        cache_record_db_cln = cache_record_db.copy(deep=True)

        # Define subsection based on `cancer_img_columns`
        cancer_img_subsection = cache_record_db_cln[cancer_img_columns]

        # Add a column to allow the user to identify the API which provided the data
        cancer_img_subsection['source_api'] = ['tcia'] * cancer_img_subsection.shape[0]

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

        Standardize interfaces.
        Note: classes are assumed to have a class attr called ``cache_record_db``.
        
        yields a single dataframe with the following columns:
        
         - 'image_id'
         - 'image_caption'
         - 'modality_best_guess'
         - 'age'
         - 'sex'
         - 'disease'
         - 'query'
         - 'query_time'
         - 'harvest_success'
         - 'files_path'
         - 'source_api'

        :param interfaces: instances of: ``OpeniInterface``, ``CancerImageInterface`` or both inside a tuple. 
        :rtype interfaces: ``tuple``, ``list``, ``OpeniInterface`` class or ``CancerImageInterface`` class.
        :return: standardize interfaces
        :rtype: ``Pandas DataFrame``
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

    Integration of Disease Ontology data.

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
                            'disease_synonym': tuple or None,
                            'diagnosis_definition' ('def'): str or None},
                  ...}``

        - the keys of the nested dictionaries in ``ont_name_dict``

        - a dictionary with 'disease_synonym' as keys and related names in a list:
          Form ``{'disease_synonym': ['name', 'name', ...], ...}``

        :param ontology_df: yield of ``DiseaseOntInterface().pull()``
        :type ontology_df: ``Pandas DataFrame``
        :return: see method description.
        :rtype: ``tuple``
        """
        # Init
        ont_name_dict, ont_disease_synonym_dict = dict(), dict()
        ont_name_dict_nest_keys = ('disease_family', 'disease_synonym', 'disease_definition')

        def str_split(s, split_on='; '):
            return tuple(s.split('; ')) if isinstance(s, str) else s

        for name, is_a, disease_synonym, defn in zip(*[ontology_df[c] for c in ('name', 'is_a', 'synonym', 'def')]):
            disease_synonym_split = str_split(disease_synonym)
            if not items_null(name):
                # Update `ont_name_dict`
                ont_name_dict[name] = {'disease_family': str_split(is_a),
                                       'disease_synonym': disease_synonym_split,
                                       'disease_definition': defn}

                # Update `ont_disease_synonym_dict`
                if isinstance(disease_synonym_split, tuple):
                    for s in disease_synonym_split:
                        if s not in ont_disease_synonym_dict:
                            ont_disease_synonym_dict[s] = [name]
                        # Check a duplicate is not added under a given disease_synonym
                        elif name not in ont_disease_synonym_dict[s]:
                            ont_disease_synonym_dict[s] += [name]

        return ont_name_dict, ont_name_dict_nest_keys, {k: sorted(v) for k, v in ont_disease_synonym_dict.items()}

    def __init__(self, cache_path=None, verbose=True):
        self.verbose = verbose
        # Load the databse
        ontology_df = DiseaseOntInterface(cache_path=cache_path, verbose=verbose).pull()

        # Obtain dictionaries
        self.ont_name_dict, self.ont_name_dict_nest_keys, self.ont_disease_synonym_dict = self._dis_ont_dict_gen(ontology_df)

        # Conver `ont_name_dict_nest_keys` to an empty dict.
        self.empty_nest_dict = dict.fromkeys(self.ont_name_dict_nest_keys, np.NaN)

        # Extract keys from the two dictionaries passed
        self.ont_name_dict_keys = tuple(self.ont_name_dict.keys())
        self.ont_disease_synonym_dict_keys = tuple(self.ont_disease_synonym_dict.keys())

    def _disease_synonym_match(self, disease_synonym):
        """

        Maps a disease synonym to an *actual* disease name.
        The 'disease_synonym' key of the dictionary which is returned will have `disease_synonym` removed
        and will have a disease names which are mapped to `disease_synonym` installed in its place.
        
        Put another way, `ont_name_dict` gives the formal name. If we have a disease which is not in 
        this dictionary, we may find it in a list of synonyms associated with that disease.

        :param disease_synonym: a disease synonym.
        :param disease_synonym: ``str``
        :return: data for a disease which the input `disease_synonym` is a synonym.
        :rtype: ``dict``
        """
        # Mapping from disease_synonym to related diseases
        ont_dis_names = self.ont_disease_synonym_dict[disease_synonym]

        # Simply use the first disease name related to the disease_synonym.
        # Note: this *assumes* that which 'name' is chosen from the list is irrelevant.
        # If the disease ontology database is not consistant, this assumption is invalid.
        disease_info = deepcopy(self.ont_name_dict[ont_dis_names[0]])
        # Remove the synonym from the 'disease_synonym' key and add 'ont_dis_names'
        if isinstance(disease_info['disease_synonym'], tuple):
            disease_synonym_new = [i for i in disease_info['disease_synonym'] if i != disease_synonym] + ont_dis_names
        else:
            disease_synonym_new = [ont_dis_names]

        # Add to `disease_info` (and remove any possible duplicates)
        disease_info['disease_synonym'] = tuple(sorted(set(disease_synonym_new)))

        return disease_info

    def _find_disease_info_raw(self, disease):
        """

        Try to match the input (`disease`) to information in the Disease Ontology Database.

        :param disease: a disease name.
        :param disease: ``str``
        :return: information on the disease (see ``_dis_ont_dict_gen()``).
        :rtype: ``dict`` or ``None``
        """
        if items_null(disease) or disease is None:
            return None
        elif disease in self.ont_name_dict:
            return self.ont_name_dict[disease]
        elif disease in self.ont_disease_synonym_dict:
            return self._disease_synonym_match(disease_synonym=disease)
        else:
            return None

    def _find_disease_info(self, disease, fuzzy_threshold):
        """
        
        Look up the family, synonyms and definition for a given ``disease``.

        :param disease: a disease name.
        :param disease: ``str``
        :param fuzzy_threshold: an integer on ``(0, 100]``.
        :type fuzzy_threshold: ``int``, `bool`, ``None``
        :return: disease information dictionary.
        :rtype: ``dict``
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

        # Try using `ont_disease_synonym_dict`
        disease_synonym_fuzzy_match, threshold = process.extractOne(disease, self.ont_disease_synonym_dict_keys)
        if threshold >= fuzzy_threshold:
            return self._disease_synonym_match(disease_synonym_fuzzy_match)
        else:
            return self.empty_nest_dict

    def integration(self, data_frame, fuzzy_threshold=False):
        """

        Look up the 'disease_family', 'disease_synonym' and 'disease_definition' () colums to ``data_frame``
        using Disease Ontology data.

        :param data_frame: a dataframe which has been passed through ``_ImagesInterfaceIntegration().integration()``
        :type data_frame: ``Pandas DataFrame``
        :param fuzzy_threshold: an integer on ``(0, 100]``.
        :type fuzzy_threshold: ``int``, `bool`, ``None``
        :return: ``data_frame`` with the columns enumerated in the description.
        :rtype: ``Pandas DataFrame``
        """
        if fuzzy_threshold is True:
            raise ValueError("`fuzzy_threshold` cannot be `True`. Please provide a specific integer on ``(0, 100]``.")

        if self.verbose:
            header("Integrating Disease Ontology Data... ")

        # Extract disease information using the Disease Ontology database
        disease_ontology_data = data_frame['disease'].progress_map(
            lambda x: self._find_disease_info(x, fuzzy_threshold)
        )

        # Convert `disease_ontology_data` to a dataframe
        disease_ontology_addition = pd.DataFrame(disease_ontology_data.tolist())

        # Add the columns in `disease_ontology_addition` to `data_frame`.
        for c in self.ont_name_dict_nest_keys:
            data_frame[c] = disease_ontology_addition[c]

        return data_frame


# ----------------------------------------------------------------------------------------------------------
# Tools to Add Data by Matching Against Disease and Disease disease_synonyms.
# ----------------------------------------------------------------------------------------------------------


def _disease_synonym_match_battery(disease, disease_synonyms, resource_dict, fuzzy_threshold):
    """

    Try to match ``disease`` and ``disease_synonyms`` in ``resource_dict``
    and return the corresponding nested dictionary (i.e., value).

    :param disease: a disease name.
    :param disease: ``str``
    :param disease_synonyms: synonyms for ``disease``
    :type disease_synonyms: ``tuple``
    :param resource_dict: a nested dictionary (see: ``_DiseaseOntologyIntegration()._dis_ont_dict_gen()``).
    :type resource_dict: ``dict``
    :param fuzzy_threshold: an integer on ``(0, 100]``.
    :type fuzzy_threshold: ``int``, `bool`, ``None``
    :return: the nested dictionary for a given key.
    :rtype: ``dict`` or ``None``
    """
    # Extract the keys
    lookup_dict_keys = tuple(resource_dict.keys())

    # Try disease 'as is'
    if disease in resource_dict:
        return resource_dict[disease]

    # Search through disease_synonyms
    if isinstance(disease_synonyms, tuple):
        for s in disease_synonyms:
            if s in resource_dict:
                return resource_dict[s]

    # Eject if fuzzy matching is disabled
    if not is_int(fuzzy_threshold):
        return np.NaN

    # Try Fuzzy matching on `disease`
    disease_fuzzy_match, threshold = process.extractOne(disease, lookup_dict_keys)
    if threshold >= fuzzy_threshold:
        return resource_dict[disease_fuzzy_match]

    # Try Fuzzy matching on `disease_synonyms`
    if not isinstance(disease_synonyms, tuple):
        return np.NaN
    else:
        for s in disease_synonyms:
            disease_synonym_fuzzy_match, threshold = process.extractOne(s, lookup_dict_keys)
            if threshold >= fuzzy_threshold:
                return resource_dict[disease_synonym_fuzzy_match]
        else:
            return np.NaN  # capitulate


def _resource_integration(data_frame, resource_dict, fuzzy_threshold, new_column_name):
    """

    Integrates information in ``resource_dict`` into ``data_frame`` as new column (``new_column_name``).
    
    :param data_frame: a dataframe which has been passed through ``_DiseaseOntologyIntegration().integration()``
    :type data_frame: ``Pandas DataFrame``
    :param resource_dict: a nested dictionary (see: ``_DiseaseOntologyIntegration()._dis_ont_dict_gen()``).
    :type resource_dict: ``dict``
    :param fuzzy_threshold: an integer on ``(0, 100]``.
    :type fuzzy_threshold: ``int``, `bool`, ``None``
    :param new_column_name: the name of the column with the extracted information.
    :type new_column_name: ``str``
    :return: ``data_frame`` with information extracted from ``resource_dict``
    :rtype: ``Pandas DataFrame``
    """
    if fuzzy_threshold is True:
        raise ValueError("`fuzzy_threshold` cannot be `True`. Please specify a specific integer on ``(0, 100]``.")

    missing_column_error_message = "`data_frame` must contain a '{0}' column.\n" \
                                   "Call ``_DiseaseOntologyIntegration().disease_ont_integration()``"

    if 'disease' not in data_frame.columns:
        raise AttributeError(missing_column_error_message.format('disease'))
    elif 'disease_synonym' not in data_frame.columns:
        raise AttributeError(missing_column_error_message.format('disease_synonym'))

    # Map gene-disease information onto the dataframe
    rslt = data_frame.progress_apply(lambda x: _disease_synonym_match_battery(disease=x['disease'],
                                                                              disease_synonyms=x['disease_synonym'],
                                                                              resource_dict=resource_dict,
                                                                              fuzzy_threshold=fuzzy_threshold), axis=1)

    # Add the `rslt` series to `data_frame`
    data_frame[new_column_name] = rslt

    return data_frame


# ----------------------------------------------------------------------------------------------------------
# Disease Symptoms Interface (Symptomatology)
# ----------------------------------------------------------------------------------------------------------


class _DiseaseSymptomsIntegration(object):
    """

    Integration of Disease Symptoms information.

    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
    Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    """

    def _disease_symptom_dict_gen(self, dis_symp_db):
        """

        Tool to create a dictionary mapping disease to symptoms.

        :param dis_symp_db: yield of ``DiseaseSymptomsInterface().pull()``
        :type dis_symp_db: ``Pandas DataFrame``
        :return: a dictionary of the form ``{disease name: [symptom, symptom, symptom, ...], ...}``
        :rtype: ``dict``
        """
        d = defaultdict(set)
        for disease, symptom in zip(dis_symp_db['common_disease_name'], dis_symp_db['common_symptom_term']):
            d[disease.lower()].add(symptom.lower())
        return {k: tuple(sorted(v)) for k, v in d.items()}

    def __init__(self, cache_path=None, verbose=True):
        self.verbose = verbose
        # Load the Disease Symptoms database
        dis_symp_db = DiseaseSymptomsInterface(cache_path=cache_path, verbose=verbose).pull()

        # Create a disease-symptoms mapping
        self.disease_symptom_dict = self._disease_symptom_dict_gen(dis_symp_db)

    def integration(self, data_frame, fuzzy_threshold=False):
        """

        Adds a 'known_associated_symptoms' column to ``data_frame`` based on the Disease Symptoms database.

        :param data_frame: a dataframe which has been passed through ``_DiseaseOntologyIntegration().integration()``
        :type data_frame: ``Pandas DataFrame``
        :param fuzzy_threshold: an integer on ``(0, 100]``.
        :type fuzzy_threshold: ``int``, ``bool``, ``None``
        :return: ``data_frame`` with a 'known_associated_symptoms' column.
        :rtype: ``Pandas DataFrame``
        """
        if self.verbose:
            header("Integrating Disease Symptoms Data... ")

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

        Generates a dictionary of the form: ``{disease name: (gene name, disgenet score), ...}``

        :param disgenet_df: yield of ``DisgenetInterface().pull('all)``.
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
        self.verbose = verbose
        # Load the database
        disgenet_df = DisgenetInterface(cache_path=cache_path, verbose=verbose).pull('all')

        # Extract the relevant information in `disgenet_df` to a dictionary.
        self.disease_gene_dict = self._disease_gene_dict_gen(disgenet_df)

    def integration(self, data_frame, fuzzy_threshold=False):
        """

        Adds a series of genes known to be associated with the given disease to ``data_frame``.

        :param data_frame: a dataframe which has been passed through ``_DiseaseOntologyIntegration().integration()``
        :type data_frame: ``Pandas DataFrame``
        :param fuzzy_threshold: an integer on ``(0, 100]``.
        :type fuzzy_threshold: ``int``, `bool`, ``None``
        :return: ``data_frame`` with a 'known_associated_genes' column.
        :rtype: ``Pandas DataFrame``
        """
        if self.verbose:
            header("Integrating DisGeNET Data... ")

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

    """
    try:
        from fuzzywuzzy import process
        global process
    except ImportError:
        error_msg = "`fuzzy_threshold` requires the `fuzzywuzzy` library,\n" \
                    "which can be installed with `$ pip install fuzzywuzzy`.\n" \
                    "For best performance, it is also recommended that python-Levenshtein is installed.\n" \
                    "(`pip install python-levenshtein`)."
        raise ImportError(error_msg)


def unify(interfaces, cache_path=None, verbose=True, fuzzy_threshold=False):
    """

    Tool to unify image interfaces (``OpeniInterface`` and ``CancerImageInterface``)
    with Diagnostic and Genomic Data.

    :param interfaces: instances of: ``OpeniInterface``, ``CancerImageInterface`` or both inside a list.
    :type interfaces: ``list``, ``tuple``, ``OpeniInterface`` or ``CancerImageInterface``.
    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
                       Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    :param fuzzy_threshold: an integer on ``(0, 100]``. If ``True`` a threshold of `95` will be used. Defaults to ``False``.
    
                .. warning::

                        Fuzzy searching with large databases, such as those this function integrates, is very
                        computationally expensive.
    
    :type fuzzy_threshold: ``int``, ``bool``, ``None``
    :return: a dataframe which unifies image interfaces with  genomic and diagnostics data.
    :rtype: ``Pandas DataFrame``

    :Example:

    >>> from biovida.images.openi_interface import OpeniInterface
    >>> from biovida.images.cancer_image_interface import CancerImageInterface
    ...
    >>> opi = OpeniInterface()
    >>> cii = CancerImageInterface(YOUR_API_KEY_HERE)
    ...
    >>> udf = unify(interfaces=[opi, cii])
    """
    # Catch ``fuzzy_threshold=True`` and set to a reasonably high default.
    if fuzzy_threshold is True:
        fuzzy_threshold = 95

    if is_int(fuzzy_threshold):
        _try_fuzzywuzzy_import()

    # Combine Instances
    combined_df = _ImagesInterfaceIntegration().integration(interfaces=interfaces)

    # Disease Ontology
    combined_df = _DiseaseOntologyIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    # Disease Symptoms
    combined_df = _DiseaseSymptomsIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    # Disgenet
    combined_df = _DisgenetIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    return combined_df























