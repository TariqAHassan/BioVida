"""

    Unify BioVida APIs
    ~~~~~~~~~~~~~~~~~~

"""
# Imports
from biovida.support_tools.support_tools import is_int

# Subpackage Unification Tools -- Images
from biovida.images._unify_images_against_other_biovia_apis import (_ImagesInterfaceIntegration,
                                                                    _DiseaseOntologyIntegration,
                                                                    _DiseaseSymptomsIntegration,
                                                                    _DisgenetIntegration)


# ----------------------------------------------------------------------------------------------------------
# Unify Against Images
# ----------------------------------------------------------------------------------------------------------


def unify_against_images(interfaces, cache_path=None, verbose=True, fuzzy_threshold=False):
    """

    Tool to unify image interfaces (``OpeniInterface`` and ``CancerImageInterface``)
    with Diagnostic and Genomic Data.

    :param interfaces: instances of: ``OpeniInterface``, ``CancerImageInterface`` or both inside a list.
    :type interfaces: ``list``, ``tuple``, ``OpeniInterface`` or ``CancerImageInterface``
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
        * 'known_associated_genes'

    .. warning::

        The `'known_associated_symptoms'` and `'known_associated_genes'` columns denote symptoms and genes
        known to be associated with the disease the patient presented with. **These columns are not an account
        of the symptomatology or genotype of the individual patient**.

    :Example:

    >>> from biovida.unify_domains import unify_against_images
    >>> from biovida.images.openi_interface import OpeniInterface
    >>> from biovida.images.cancer_image_interface import CancerImageInterface
    ...
    >>> opi = OpeniInterface()
    # --- Search and Pull ---
    >>> cii = CancerImageInterface(YOUR_API_KEY_HERE)
    # --- Search and Pull ---
    ...
    >>> udf = unify_against_images(interfaces=[opi, cii])
    """
    # Catch ``fuzzy_threshold=True`` and set to a reasonably high default.
    if fuzzy_threshold is True:
        fuzzy_threshold = 95

    # Combine Instances
    combined_df = _ImagesInterfaceIntegration().integration(interfaces=interfaces)

    # Disease Ontology
    combined_df = _DiseaseOntologyIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    # Disease Symptoms
    combined_df = _DiseaseSymptomsIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    # Disgenet
    combined_df = _DisgenetIntegration(cache_path, verbose).integration(combined_df, fuzzy_threshold)

    return combined_df

























