Project Overview
----------------

This library is primarily intended to help solve an outstanding problem in biomedical data science: a lack of usable data.
While impressive data mining work in bioinformatics in recent years has helped build clean databases of known gene-disease
associations, a poverty of machine learning ready biomedical images persists. This is partly because cleaning datasets
of biomedical images is very complex -- so complex in fact, it must often be done by hand. This package is an attempt
at automating this process. This is done in two main ways. First, using standard programmatic techniques to
harvest data from online databases. Second, to clean some forms of data (images namely), machine learning itself is used to
identify properties which are liable to corrupt the dataset (e.g., large watermarks which obscure an image).
Steps can then be taken correct or remove this problematic data.

While BioVida is currently focused on harvesting and processing biomedical images, it contains (or will contain)
tools to perform analogous tasks with other types of data (namely genomics and disease diagnostics).
For this reason BioVida has modular structure, with different types of biomedical data handled by distinct subpackages
within `biovida`.

--------------

Installation
------------

Latest Build:

.. code:: bash

    $ pip install git+git://github.com/TariqAHassan/BioVida@master

--------------

Dependencies
------------

BioVida requires: `inflect <https://pypi.python.org/pypi/inflect>`__,
`pandas <http://pandas.pydata.org>`__, `numpy <http://www.numpy.org>`__,
`requests <http://docs.python-requests.org/en/master/>`__ and
`tqdm <https://github.com/tqdm/tqdm>`__.

--------------

Image Data
----------

Import the Interface for the NIH's Open-i API.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from biovida.images.openi_interface import OpenInterface

Create an Instance of the Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    opi = OpenInterface()

Perform a Search
^^^^^^^^^^^^^^^^

.. code:: python

    opi.search("caudate nucleus", image_type=['mri', 'pet', 'ct'])
    # Results Found: 1,165.

The values accepted by the ``image_type`` argument above can easily be
reviewed:

.. code:: python

    opi.options(search_parameter='image_type')

Additionally, searches can easily be reviewed:

.. code:: python

    opi.current_search
    # {'image_type': ['mri', 'pet', 'ct', 'exclude_graphics'], 'query': 'caudate nucleus'}

    opi.current_search_total
    # 1165

Pull the data
^^^^^^^^^^^^^

.. code:: python

    df = opi.pull()

The DataFrame created above, ``df``, contains data from all fields
provided by the Open-i API.† Images referenced in the DataFrame will
automatically be harvested (unless specified otherwise).

†\ *Note*: by default, data harvesting is truncated after the first 60
results.

--------------

Genomic Data
------------

Import the Interface for DisGeNET.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from biovida.genomics.disgenet_interface import DisgenetInterface

Create an Instance of the Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    dna = DisgenetInterface()

Options: Explore Available Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    dna.options()
    # Available Databases:
    #   - 'all'
    #   - 'curated'
    #   - 'snp_disgenet'

    dna.options('curated')
    # - Full Name:    Curated Gene-Disease Associations
    # - Description:  The file contains gene-disease associations from UNIPROT, CTD (human subset),
    #                 ClinVar, Orphanet, and the GWAS Catalog.

Pull the data
^^^^^^^^^^^^^

.. code:: python

    df = dna.pull('curated')

This database will be cached to allow to fast access in the future.

As with the ``OpenInterface()`` class above, it is easy to gain access
to the most recent ``pull`` and related information.

The database its self:

.. code:: python

    dna.current_database

Information about the database:

.. code:: python

    dna.current_database_name
    # 'curated'

    dna.current_database_full_name
    # 'Curated Gene-Disease Associations'

    dna.current_database_description
    # 'The file contains gene-disease associations from...'


Resources
---------

Images

-  The `Open-i <https://openi.nlm.nih.gov>`__ BioMedical Image Search
   Engine (NIH)

Genomics

-  `DisGeNET <http://www.disgenet.org/web/DisGeNET/menu>`__

   -  Janet Piñero, Àlex Bravo, Núria Queralt-Rosinach, Alba
      Gutiérrez-Sacristán, Jordi Deu-Pons, Emilio Centeno, Javier
      García-García, Ferran Sanz, and Laura I. Furlong. DisGeNET: a
      comprehensive platform integrating information on human
      disease-associated genes and variants. Nucl. Acids Res. (2016)
      doi:10.1093/nar/gkw943

   -  Janet Piñero, Núria Queralt-Rosinach, Àlex Bravo, Jordi Deu-Pons,
      Anna Bauer-Mehren, Martin Baron, Ferran Sanz, Laura I. Furlong.
      DisGeNET: a discovery platform for the dynamical exploration of
      human diseases and their genes. Database (2015)
      doi:10.1093/database/bav028
