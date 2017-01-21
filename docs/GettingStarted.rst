Getting Started
---------------

This library is primarily intended to help solve an outstanding problem in biomedical data science: a lack of usable data.
While impressive data mining work in bioinformatics in recent years has helped build clean databases of known gene-disease
associations, a poverty of machine learning ready biomedical images persists. BioVida is designed to automate this process,
reducing, if not eliminating, the need for data cleaning, freeing you up to focus on data analysis itself.

While focused on image data, BioVida is able to curate a broad range of biomedical data including diagnostics and genomics.
Some of this is accomplished using standard programming techniques, the rest using neural networks.
It is important to note that the neural networks are intended to operate seamlessly *behind the scenes*.
Standard use of the library will never bring you into direct contact with anything other than their output.

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
`scipy <https://www.scipy.org>`__,
`scikit-image <http://scikit-image.org>`__ and
`keras <https://keras.io>`__


All of these dependencies should be installed automatically when installing BioVida.

**Note**: Keras is used to power the Convolutional Neural Networks used in this project, meaning
either `TensorFlow <https://www.tensorflow.org>`__ or
`Theano <http://deeplearning.net/software/theano/>`__ can be used as a computational backend when using BioVida.
If neither is present at install time, BioVida will automatically install TensorFlow for you.

--------------

Image Data
----------

Import and Instantiate the Open-i Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from biovida.images.openi_interface import OpenInterface

    opi = OpenInterface()

Perform a Search
^^^^^^^^^^^^^^^^

.. code:: python

    opi.search("aneurysm", image_type=['mri', 'ct'])
    # Results Found: 3,973.

The values accepted by the ``image_type`` argument above can easily be
reviewed:

.. code:: python

    opi.options(search_parameter='image_type')

Additionally, searches can easily be reviewed:

.. code:: python

    opi.current_search
    # {'image_type': ['mri', 'ct', 'exclude_graphics'], 'query': 'aneurysm'}

    opi.current_search_total
    # 3973

Pull the data
^^^^^^^^^^^^^

.. code:: python

    search_df = opi.pull()

The DataFrame created above, ``df``, contains data from all fields
provided by the Open-i API.† Images referenced in the DataFrame will
automatically be harvested (unless specified otherwise).

†\ *Note*: by default, data harvesting is truncated after the first 60
results.


Automated Image Data Cleaning
-----------------------------

Cleaning the downloaded images is extremely simple.


Import the ImageProcessing Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from biovida.images.image_processing import ImageProcessing


# Instantiate the Tool using the OpenInterface Instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    ip = ImageProcessing(opi)


Clean the Image Data
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    cdf = ip.auto()


Save the Cleaned Images
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    ip.save("/save/directory/")


That's it.


While the `ImageProcessing` classes allows you to
to control the image processing more precisely if you
wish (see the documentation `here <https://tariqahassan.github.io/BioVida/API.html#image-processing>`__), this
fully automated approach should suffice in most cases.

--------------

Genomic Data
------------

Import the Interface for DisGeNET
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from biovida.genomics.disgenet_interface import DisgenetInterface

Create an Instance of the Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    dna = DisgenetInterface()

Explore Available Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
