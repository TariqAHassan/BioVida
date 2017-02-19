Getting Started
---------------

This library is primarily intended to automate the collection, post-processing and integration of biomedical data stored
in public online databases. It is hoped that this effort will catalyze new insights and understanding by transforming
multiple, distinct data repositories into unified datasets which are highly amenable to data analysis.

BioVida aims to curate a broad range of biomedical information. In areas such as diagnostics and genomics, this
involves drawing on the work of others, such as the impressive work by the Disease Ontology and DisGeNET teams.
In the case of image data however, BioVida itself performs the 'heavy lifting' of preparing raw data from online
sources. This is made possible by combining traditional software development approaches with recent advances
in machine learning, namely convolutional neural networks.

The guide below provides a brief introduction to getting started with BioVida.

--------------

Installation
------------

Latest Build:

.. code:: bash

    $ pip install git+git://github.com/TariqAHassan/BioVida@master

--------------

Dependencies
------------

BioVida requires: `pandas <http://pandas.pydata.org>`__,
`numpy <http://www.numpy.org>`__,
`requests <http://docs.python-requests.org/en/master/>`__,
`tqdm <https://github.com/tqdm/tqdm>`__,
`pillow <https://github.com/python-pillow/Pillow>`__,
`pydicom <https://github.com/darcymason/pydicom>`__,
`h5py <http://www.h5py.org>`__,
`scipy <https://www.scipy.org>`__,
`scikit-image <http://scikit-image.org>`__ and
`keras <https://keras.io>`__

All of these dependencies should be installed automatically when installing BioVida.

**Notes**:

1. Keras is used to power the convolutional neural networks in this project. This has the advantage of
allowing either `TensorFlow <https://www.tensorflow.org>`__ or
`Theano <http://deeplearning.net/software/theano/>`__ to be used as a computational backend.
If neither is present at install time, BioVida will automatically install TensorFlow for you.

2. To use ``scipy`` on macOS (formerly OSX) you will need ``gcc`` and ``pillow``, which
can be obtained with ``$ brew install gcc`` and ``$ pip install pillow``, respectively.
If you do not have ``homebrew`` installed (needed for ``brew``), you can download it `here <https://brew.sh>`__.

--------------

Image Data
----------

Cancer Imaging Archive
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # 1. Import the interface for the Cancer Imaging Archive
    from biovida.images.cancer_image_interface import CancerImageInterface

    # 2. Create an Instance of the Tool
    cii = CancerImageInterface(YOUR_API_KEY_HERE)

    # 3. Perform a search
    cii.search(location='extremities')

    # 4. Pull the data
    cdf = cii.pull()

Open-i BioMedical Image Search Engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # 1. Import the Interface for the NIH's Open-i API.
    from biovida.images.openi_interface import OpeniInterface

    # 2. Create an Instance of the Tool
    opi = OpeniInterface()

    # 3. Perform a general search for MRIs and CTs
    opi.search(query=None, image_type=['mri', 'ct'])  # Results Found: 134,113.

    # 4. Pull the data
    search_df = opi.pull()

The DataFrame created above, ``df``, contains data from all fields
provided by the Open-i API. :superscript:`†` Images referenced in the DataFrame will
automatically be harvested (unless specified otherwise).

The values accepted by the ``image_type`` argument above can easily be
reviewed:

.. code:: python

    opi.options(search_parameter='image_type')

Additionally, searches can easily be reviewed:

.. code:: python

    opi.current_search
    # {'image_type': ['mri', 'ct', 'exclude_graphics'], 'query': ''}

    opi.current_search_total
    # 134113

:superscript:`†` *Note:* by default, data harvesting is truncated after the first 60
results.

Automated Image Data Cleaning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cleaning the images which have been downloaded is extremely simple.

.. code:: python

    # 1. Import Image Processing Tools
    from biovida.images.image_processing import ImageProcessing

    # 2. Instantiate the Tool using the OpeniInterface Instance
    ip = ImageProcessing(opi)

    # 3. Clean the Image Data
    idf = ip.auto()

    # 4. Save the Cleaned Images
    ip.save("/save/directory/")

While the ``ImageProcessing()`` classes allows you to
to control the image processing more precisely if you
wish (see the documentation `here <https://tariqahassan.github.io/BioVida/API.html#image-processing>`__), this
fully automated approach should suffice in most cases.

**Notice**: This library is still in *pre-alpha*. That is, formal unit testing has not yet been implemented.
**Until it is, this software should be considered to be experimental**.

--------------

Genomic Data
------------

Data Harvesting
^^^^^^^^^^^^^^^

.. code:: python

    # 1. Create an instance of the tool
    from biovida.genomics.disgenet_interface import DisgenetInterface

    # 2. Create an Instance of the Tool
    dna = DisgenetInterface()

    # 3. Pull the data
    df = dna.pull('curated')


Exploring Available Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

This database will be cached to allow to fast access in the future.

As with the ``OpeniInterface()`` class above, it is easy to gain access
to the most recent ``pull`` and related information.

The database itself:

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

--------------

Diagnostic Data
---------------

Data Harvesting
^^^^^^^^^^^^^^^

.. code:: python

    # 1. Import the Interface for DiseaseOntology.org
    from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface

    # 2. Create an Instance of the Tool
    doi = DiseaseOntInterface()

    # 3. Pull the Database
    ddf = doi.pull()

One can gain access to the database, by following
the approach shown above (with ``ddf``) or as follows:

.. code:: python

    doi.disease_db

It is also possible to inspect the date on which
the database was created by *DiseaseOntology.org:*

.. code:: python

    doi.db_date
    # datetime.datetime(2017, 1, 13, 0, 0)

--------------

Resources
---------

Images

-  The `Cancer Imaging Archive <http://www.cancerimagingarchive.net>`__

-  The `Open-i <https://openi.nlm.nih.gov>`__ BioMedical Image Search Engine (NIH)

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

Diagnostics

- `DiseaseOntology <http://disease-ontology.org/>`__
