"""

    Unifying BioVida APIs
    ~~~~~~~~~~~~~~~~~~~~~

"""
# Subpackage Unification Tools -- Images
from biovida.images._unify_images_against_other_biovida_apis import images_unify


# ----------------------------------------------------------------------------------------------------------
# Unify Against Images
# ----------------------------------------------------------------------------------------------------------


def unify_against_images(interfaces, cache_path=None, verbose=True, fuzzy_threshold=False):
    """

    Tool to unify image interfaces (namely ``OpeniInterface`` and/or ``CancerImageInterface``)
    with Diagnostic and Genomic Data.

    Note: this tool extracts the ``cache_records_db`` attribute from these classes.

    :param interfaces: instances of: ``OpeniInterface``, ``CancerImageInterface`` or both inside a list.
    :type interfaces: ``list``, ``tuple``, ``OpeniInterface`` or ``CancerImageInterface``
    :param cache_path: location of the BioVida cache. If one does not exist in this location, one will created.
                       Default to ``None`` (which will generate a cache in the home folder).
    :type cache_path: ``str`` or ``None``
    :param verbose: If ``True``, print notice when downloading database. Defaults to ``True``.
    :type verbose: ``bool``
    :param fuzzy_threshold: an integer on ``(0, 100]``. If ``True`` a threshold of `95` will be used. Defaults to ``False``.
    
                .. warning::

                        While this parameter will likely increase the number of matches, fuzzy searching with
                        large databases, such as those this function integrates, is very computationally expensive.
    
    :type fuzzy_threshold: ``int``, ``bool``, ``None``
    :return: a dataframe which unifies image interfaces with  genomic and diagnostics data.
    :rtype: ``Pandas DataFrame``


    This function evolves a DataFrame with the following columns:

    .. hlist::
        :columns: 4

        * 'image_id'
        * 'image_caption'
        * 'modality_best_guess'
        * 'age'
        * 'sex'
        * 'disease'
        * 'query'
        * 'pull_time'
        * 'harvest_success'
        * 'files_path'
        * 'source_api'
        * 'disease_family'
        * 'disease_synonym'
        * 'disease_definition'
        * 'known_associated_symptoms'
        * 'patient_symptoms'
        * 'known_associated_genes'


    .. note::

        The `'known_associated_genes'` column is of the form ``((Gene Name, DisGeNET Evidence Score), ...)``.

    .. warning::

        The `'known_associated_symptoms'` and `'known_associated_genes'` columns denote symptoms and genes
        known to be associated with the disease the patient presented with. **These columns are not an account
        of the symptomatology or genotype of the patients themselves**. Conversely, 'patient_symptoms' *is*
        an account of a given patient's symptoms.

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
    return images_unify(interfaces=interfaces, cache_path=cache_path, verbose=verbose, fuzzy_threshold=fuzzy_threshold)




















