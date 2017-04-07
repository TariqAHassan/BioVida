# coding: utf-8

"""

    Unifying Across Domains
    ~~~~~~~~~~~~~~~~~~~~~~~

"""
# Subpackage Unification Tools -- Images
from biovida.images._unify_images_against_other_biovida_apis import images_unify


# ----------------------------------------------------------------------------------------------------------
# Unify Against Images
# ----------------------------------------------------------------------------------------------------------


def unify_against_images(instances,
                         db_to_extract='records_db',
                         verbose=True,
                         fuzzy_threshold=False):
    """

    Tool to unify image instances (namely ``OpeniInterface`` and/or ``CancerImageInterface``)
    with Diagnostic and Genomic Data.

    :param instances: any one of ``OpeniInterface``, ``CancerImageInterface`` or ``OpeniImageProcessing``, or some
                           combination inside an iterable.
    :type instances: ``list``, ``tuple``, ``OpeniInterface``, ``CancerImageInterface`` or ``OpeniImageProcessing``.
    :param db_to_extract: the database to use. Must be one of: 'records_db', 'cache_records_db' or 'image_dataframe'.
                          Defaults to 'records_db'.

                    .. note::

                        If an instance of ``OpeniImageProcessing`` is passed to ``instances``, the ``image_dataframe``
                        attribute will be extracted regardless of the value passed to this argument.

    :type db_to_extract: ``str``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    :param fuzzy_threshold: an integer on ``(0, 100]``. If ``True`` a threshold of `95` will be used. Defaults to ``False``.
    
                .. warning::

                        While this parameter will likely increase the number of matches, fuzzy searching with
                        large databases, such as those this function integrates, is very computationally expensive.
    
    :type fuzzy_threshold: ``int``, ``bool``, ``None``
    :return: a dataframe which unifies image instances with genomic and diagnostics data.
    :rtype: ``Pandas DataFrame``


    This function evolves a DataFrame with the following columns:

    .. hlist::
        :columns: 4

        * 'age'
        * 'article_type'
        * 'disease'
        * 'image_caption'
        * 'image_id'
        * 'modality_best_guess'
        * 'pull_time'
        * 'query'
        * 'sex'
        * 'source_api'
        * 'disease_family'
        * 'disease_synonym'
        * 'disease_definition'
        * 'known_associated_symptoms'
        * 'mentioned_symptoms'
        * 'known_associated_genes'


    .. note::

        The ``'known_associated_genes'`` column is of the form ``((Gene Name, DisGeNET Evidence Score), ...)``.

    .. note:: 
    
        If an instance of the ``CancerImageInterface`` class is passed, a ``'source_images_path'``
        column will also be generated if the database extracted from the instance contains 
        a ``'cached_dicom_images_path'`` column.

    .. warning::

        The ``'known_associated_symptoms'`` and ``'known_associated_genes'`` columns denote symptoms and genes
        known to be associated with the disease the patient presented with. **These columns are not an account
        of the symptomatology or genotype of the patients themselves**. Conversely, the ``'mentioned_symptoms'``
        column is an account of a given patient's symptoms *if* the data is from a clinical case
        (i.e., ``article_type`` equals 'case_report').

    :Example:

    >>> from biovida.images import OpeniInterface
    >>> from biovida.images import CancerImageInterface
    >>> from biovida.unification import unify_against_images
    ...
    >>> opi = OpeniInterface()
    # --- Search and Pull ---
    >>> udf1 = unify_against_images(opi)
    ...
    # Adding another Interface from the images subpackage
    >>> cii = CancerImageInterface(YOUR_API_KEY_HERE)
    # --- Search and Pull ---
    >>> udf2 = unify_against_images([opi, cii])
    """
    return images_unify(instances=instances, db_to_extract=db_to_extract,
                        verbose=verbose, fuzzy_threshold=fuzzy_threshold)
