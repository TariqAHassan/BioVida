# coding: utf-8

"""

    Tools to Unify Image Data Against Other BioVida APIs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Note: To use these tools, see ``biovida.unify_domains.unify_against_images()``
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

# Import Interfaces
from biovida.genomics.disgenet_interface import DisgenetInterface
from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface
from biovida.diagnostics.disease_symptoms_interface import DiseaseSymptomsInterface

# Support Tools
from biovida.support_tools.support_tools import tqdm, is_int, items_null

# Image Tools
from biovida.images._image_tools import try_fuzzywuzzy_import

# Open-i Specific Tools
from biovida.images._interface_support.openi.openi_support_tools import possible_openi_image_processing_cols


# ----------------------------------------------------------------------------------------------------------
# Interface Integration
# ----------------------------------------------------------------------------------------------------------


class _ImagesInterfaceIntegration(object):
    """

    Tools to Unify BioVida Image Interfaces

    """

    def __init__(self):
        self._additional_columns = list()

    def _add_additional_columns(self, db, list_of_columns, skip=None):
        """

        Add ``self._additional_columns`` to ``db`` and ``list_of_columns``

        :param list_of_columns: as present in ``_open_i_prep()`` or ``_cancer_image_prep()``.
        :type list_of_columns: ``list``
        :param db: as passed to ``_open_i_prep()`` or ``_cancer_image_prep()``.
        :type db: ``Pandas DataFrame``
        :param skip: items in ``self._additional_columns`` to refrain from adding to ``db``.
        :type skip: ``list`` or ``None``
        :return: see description.
        :rtype: ``tuple``
        """
        for c in self._additional_columns:
            if skip is None or c not in skip:
                list_of_columns.append(c)
                if c not in db.columns:
                    db[c] = [None] * db.shape[0]
        return db, list_of_columns

    def _open_i_prep(self, db):
        """

        A tool to clean and standardize a database from a ``OpeniInterface`` instance.

        :param db: a database from an ``OpeniInterface`` instance.
        :type db: ``Pandas DataFrame``
        :return: a cleaned and standardize ``db``.
        :rtype: ``Pandas DataFrame``
        """
        # Deep copy the input to prevent mutating the original in memory.
        db_cln = db.copy(deep=True)

        # Column which provides a guess, based on the text, on which imaging modality created the image.
        db_cln['modality_best_guess'] = db_cln.apply(
            lambda x: x['imaging_modality_from_text'] if isinstance(x['imaging_modality_from_text'], str) else x[
                'modality_full'], axis=1)

        # Define columns to keep
        openi_columns = ['abstract', 'article_type', 'image_id_short', 'image_caption',
                         'modality_best_guess', 'age', 'sex', 'diagnosis', 'query', 'pull_time']

        openi_col_rename = {'diagnosis': 'disease'}

        # Allow for cases where images have not been downloaded.
        if 'cached_images_path' in db_cln.columns:
            openi_columns.append('cached_images_path')
            db_cln['cached_images_path'] = db_cln['cached_images_path'].map(
                lambda x: tuple([x]) if not isinstance(x, tuple) else x, na_action='ignore')

        if len(self._additional_columns):
            db_cln, openi_columns = self._add_additional_columns(db_cln, openi_columns)

        db_cln['article_type'] = db_cln['article_type'].replace({'encounter': 'case_report',
                                                                 'image_id_short': 'image_id'})

        # Define subsection based on `openi_columns`
        openi_subsection = db_cln[openi_columns]

        openi_subsection['source_api'] = ['openi'] * openi_subsection.shape[0]
        return openi_subsection.rename(columns=openi_col_rename)

    def _image_processing_prep(self, db):
        """

        A tool to clean and standardize a database from a ``OpeniImageProcessing`` instance.

        :param db: a database from a ``OpeniImageProcessing`` instance.
        :type db: ``Pandas DataFrame``
        :return: a cleaned and standardize ``db``.
        :rtype: ``Pandas DataFrame``
        """
        # Note: if the ``OpeniImageProcessing`` class is updated to handle
        # instances other than ``OpeniInterface``, this approach will need to be changed.
        return self._open_i_prep(db)

    def _cancer_image_prep(self, db):
        """

        A tool to clean and standardize a database from a ``CancerImageInterface`` instance.

        :param db: a database from a ``CancerImageInterface`` instance.
        :type db: ``Pandas DataFrame``
        :return: a cleaned and standardize ``db``.
        :rtype: ``Pandas DataFrame``
        """
        # Define columns to keep
        cancer_image_columns = ['series_instance_uid', 'series_description', 'modality_full', 'age',
                                'sex', 'article_type', 'cancer_type', 'query', 'pull_time']

        # Column name changes (based on ``_open_i_prep()``).
        cancer_image_col_rename = {'series_instance_uid': 'image_id',
                                   'series_description': 'image_caption',
                                   'cached_dicom_images_path': 'raw',
                                   'modality_full': 'modality_best_guess',
                                   'cancer_type': 'disease'}

        # Allow for cases where images have not been downloaded,
        # i.e., absent 'cached_images_path' and 'cached_dicom_images_path' columns.
        if 'cached_images_path' in db.columns:
            cancer_image_columns.append('cached_images_path')
        if 'cached_dicom_images_path' in db.columns:
            cancer_image_columns.append('cached_dicom_images_path')
            cancer_image_col_rename['cached_dicom_images_path'] = 'source_images_path'
            additional_columns_skip = 'source_images_path'
        else:
            additional_columns_skip = None

        # Deep copy the input to prevent mutating the original in memory.
        db_cln = db.copy(deep=True)

        if len(self._additional_columns):
            db_cln, cancer_image_columns = self._add_additional_columns(db=db_cln,
                                                                        list_of_columns=cancer_image_columns,
                                                                        skip=additional_columns_skip)

        # Define subsection based on `cancer_image_columns`
        cancer_image_subsection = db_cln[cancer_image_columns]

        cancer_image_subsection['abstract'] = np.NaN
        cancer_image_subsection['source_api'] = ['tcia'] * cancer_image_subsection.shape[0]
        return cancer_image_subsection.rename(columns=cancer_image_col_rename)

    @property
    def _prep_class_dict(self):
        """

        Return a dictionary which maps image interface classes to
        the methods designed to handle them.

        :return: a dictionary mapping class names to functions.
        :rtype: ``dict``
        """
        return {'OpeniInterface': self._open_i_prep,
                'CancerImageInterface': self._cancer_image_prep,
                'OpeniImageProcessing': self._image_processing_prep}

    def integration(self, instances, db_to_extract):
        """

        Standardize instances.

        This method yields a single dataframe with the following columns:

         - 'abstract'*
         - 'image_id_short'
         - 'image_caption'
         - 'modality_best_guess'
         - 'age'
         - 'sex'
         - 'disease'
         - 'query'
         - 'pull_time'
         - 'harvest_success'
         - 'files_path'
         - 'source_api'

         *NOTE: this column will be dropped after passing through ``_DiseaseSymptomsIntegration().integration()``.

        :param instances: any one of ``OpeniInterface``, ``CancerImageInterface`` or ``OpeniImageProcessing``, or some
                           combination inside an iterable.
        :type instances: ``list``, ``tuple``, ``OpeniInterface``, ``CancerImageInterface`` or ``OpeniImageProcessing``.
        :param db_to_extract: the database to use. Must be one of: 'records_db', 'cache_records_db'.
        :type db_to_extract: ``str``
        :return: standardize instances
        :rtype: ``Pandas DataFrame``
        """
        instances_types = [type(i).__name__ for i in instances]
        if 'OpeniImageProcessing' in instances_types:
            self._additional_columns += possible_openi_image_processing_cols
        if 'CancerImageInterface' in instances_types:
            self._additional_columns += ['source_images_path']

        frames = list()
        for class_instance in instances:
            interface_name = type(class_instance).__name__
            func = self._prep_class_dict[interface_name]
            if interface_name == 'OpeniImageProcessing':
                database = getattr(class_instance, "image_dataframe")
            else:
                database = getattr(class_instance, db_to_extract)
            if not isinstance(database, pd.DataFrame):
                raise ValueError("The {0} instance's '{1}' database must be of type DataFrame,\nnot "
                                 "'{2}'.".format(interface_name, db_to_extract, type(database).__name__))
            frames.append(func(database))

        self._additional_columns = list()  # reset

        combined_df = pd.concat(frames, ignore_index=True)
        combined_df['disease'] = combined_df['disease'].map(lambda x: x.lower() if isinstance(x, str) else x)

        return combined_df.fillna(np.NaN)


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

    @staticmethod
    def _dis_ont_dict_gen(ontology_df):
        """

        Convert the information obtained from ``DiseaseOntInterface().pull()`` into:

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
        ont_name_dict, ont_disease_synonym_dict = dict(), dict()
        ont_name_dict_nest_keys = ('disease_family', 'disease_synonym', 'disease_definition')

        def str_split(s, split_on='; '):
            return tuple(s.split(split_on)) if isinstance(s, str) else s

        # ToDo: change to iterrows().
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
        # Load the database
        ontology_df = DiseaseOntInterface(cache_path=cache_path, verbose=verbose).pull()

        # Obtain dictionaries
        self.ont_name_dict, self.ont_name_dict_nest_keys, self.ont_disease_synonym_dict = self._dis_ont_dict_gen(
            ontology_df)

        # Convert `ont_name_dict_nest_keys` to an empty dict.
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
        # If the disease ontology database is not consistent, this assumption is invalid.
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
        if not isinstance(disease, str):
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

        if is_int(fuzzy_threshold):
            process = try_fuzzywuzzy_import()

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

        Create the 'disease_family', 'disease_synonym' and 'disease_definition' columns to ``data_frame``
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

        # Extract disease information using the Disease Ontology database
        disease_ontology_data = [self._find_disease_info(i, fuzzy_threshold)
                                 for i in tqdm(data_frame['disease'], desc='Disease Data', disable=not self.verbose)]

        # Convert `disease_ontology_data` to a dataframe
        disease_ontology_addition = pd.DataFrame(disease_ontology_data)

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

    # Import process
    if is_int(fuzzy_threshold):
        process = try_fuzzywuzzy_import()

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


def _resource_integration(data_frame, resource_dict, fuzzy_threshold, new_column_name, verbose, desc):
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
    :param verbose: If ``True``, print notice when downloading database.
    :type verbose: ``bool``
    :param desc: description to pass to ``tqdm``.
    :type desc: ``str`` or ``None``
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
    matches = list()
    for _, row in tqdm(data_frame.iterrows(), total=len(data_frame), desc=desc, disable=not verbose):
        match = _disease_synonym_match_battery(disease=row['disease'],
                                               disease_synonyms=row['disease_synonym'],
                                               resource_dict=resource_dict,
                                               fuzzy_threshold=fuzzy_threshold)
        matches.append(match)

    # Add the `rslt` series to `data_frame`
    data_frame[new_column_name] = matches

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

    @staticmethod
    def _disease_symptom_dict_gen(dis_symp_db):
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

    def _mentioned_symptoms(self, data_frame):
        """

        Match 'known_associated_symptoms' to the 'abstract' for the given row.

        :param data_frame: ``updated_data_frame`` as evolved in ``_DiseaseSymptomsIntegration().integration()``.
        :type data_frame: ``Pandas DataFrame``
        :return: a series with tuples of 'known_associated_symptoms' found in 'abstract'.
        :rtype: ``Pandas Series``
        """
        # ToDo: consider using 'mesh' and 'problems' cols - would have to be added in ``_ImagesInterfaceIntegration()``.
        def match_symptoms(row):
            """Find items in 'known_associated_symptoms' in 'abstract'."""
            if isinstance(row['known_associated_symptoms'], (list, tuple)) and isinstance(row['abstract'], str):
                abstract_lower = row['abstract'].lower()
                symptoms = [i for i in row['known_associated_symptoms'] if i in abstract_lower]
                return tuple(symptoms) if len(symptoms) else np.NaN
            else:
                return np.NaN

        return [match_symptoms(row) for _, row in tqdm(data_frame.iterrows(),
                                                       total=len(data_frame),
                                                       desc='Matching Symptoms',
                                                       disable=not self.verbose)]

    def integration(self, data_frame, fuzzy_threshold=False):
        """

        Adds a 'known_associated_symptoms' column to ``data_frame`` based on the Disease Symptoms database.

        :param data_frame: a dataframe which has been passed through ``_DiseaseOntologyIntegration().integration()``.
        :type data_frame: ``Pandas DataFrame``
        :param fuzzy_threshold: an integer on ``(0, 100]``.
        :type fuzzy_threshold: ``int``, ``bool``, ``None``
        :return: ``data_frame`` with a 'known_associated_symptoms' column.
        :rtype: ``Pandas DataFrame``
        """
        # Generate a 'known_associated_symptoms' columns
        updated_data_frame = _resource_integration(data_frame=data_frame,
                                                   resource_dict=self.disease_symptom_dict,
                                                   fuzzy_threshold=fuzzy_threshold,
                                                   new_column_name='known_associated_symptoms',
                                                   verbose=self.verbose,
                                                   desc='Symptoms')

        # Find 'known_associated_symptoms' which individual patients presented with by scanning the abstract
        updated_data_frame['mentioned_symptoms'] = self._mentioned_symptoms(updated_data_frame)

        # Drop the 'abstract' column as it is no longer needed
        del updated_data_frame['abstract']

        return updated_data_frame


# ----------------------------------------------------------------------------------------------------------
# DisGeNET Integration
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

    @staticmethod
    def _disease_gene_dict_gen(disgenet_df):
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
        return _resource_integration(data_frame=data_frame,
                                     resource_dict=self.disease_gene_dict,
                                     fuzzy_threshold=fuzzy_threshold,
                                     new_column_name='known_associated_genes',
                                     verbose=self.verbose,
                                     desc='Genomic')


# ----------------------------------------------------------------------------------------------------------
# Unify
# ----------------------------------------------------------------------------------------------------------


def images_unify(instances, db_to_extract='records_db', verbose=True, fuzzy_threshold=False):
    """

    Unify Instances in the ``images`` subpackage against other BioVida APIs.

    :param instances: See: ``biovida.unify_domains.unify_against_images()``
    :param instances: `list``, ``tuple``, ``OpeniInterface``, ``OpeniImageProcessing`` or ``CancerImageInterface``
    :param db_to_extract: the database to use. Must be one of: 'records_db', 'cache_records_db'.
                          Defaults to 'records_db'.
    :type db_to_extract: ``str``
    :param verbose: See: ``biovida.unify_domains.unify_against_images()``
    :param verbose: ``bool``
    :param fuzzy_threshold: See: ``biovida.unify_domains.unify_against_images()``
    :param fuzzy_threshold: ``int``, ``bool``, ``None``
    :return: See: ``biovida.unify_domains.unify_against_images()``
    :rtype: ``Pandas DataFrame``
    """
    instances = [instances] if not isinstance(instances, (list, tuple)) else instances

    # Catch ``fuzzy_threshold=True`` and set to a reasonably high default.
    if fuzzy_threshold is True:
        fuzzy_threshold = 95

    # Note: this doesn't consider cases where multiple biovida
    # caches exist, each with different data, e.g., one has
    # genetic data, another has Symptoms data. Thus, in rare
    # cases, this may cause data to be downloaded unnecessarily.
    cache_path = None  # default
    for i in instances:
        if hasattr(i, '_cache_path'):
            cache_path = getattr(i, '_cache_path')
            break

    # Combine Instances
    combined_df = _ImagesInterfaceIntegration().integration(instances=instances, db_to_extract=db_to_extract)

    # Disease Ontology
    combined_df = _DiseaseOntologyIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    # Disease Symptoms
    combined_df = _DiseaseSymptomsIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    # Disgenet
    combined_df = _DisgenetIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    return combined_df
