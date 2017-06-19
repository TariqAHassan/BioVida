BioVida is a library designed to make it easy to gain access to existing
data sets of biomedical images as well as build brand new, custom-made
ones.

It is hoped that by automating the tedious data munging that is
typically involved in this process, more people will become interested
in applying machine learning to biomedical images and, in turn,
advancing insights into human disease.

In a nod to recursion, BioVida tries to accomplish some of this
automation with machine learning itself, using tools like convolutional
neural networks.

Installation
------------

Python Package Index:

.. code:: bash

    $ pip install biovida

Latest Build:

.. code:: bash

    $ pip install git+git://github.com/TariqAHassan/BioVida@master

Requires Python 3.4+

Images: Stable
--------------

In just a few lines of code, you can gain access to biomedical databases
which store tens of millions of images.

*Please note that you are bound to adhere to the copyright and other
usage restrictions under which this data is provided to you by its
creators.*

Open-i BioMedical Image Search Engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python


    # 1. Import the Interface for the NIH's Open-i API.
    from biovida.images import OpeniInterface

    # 2. Create an Instance of the Tool
    opi = OpeniInterface()

    # 3. Perform a search for x-rays and cts of lung cancer
    opi.search(query='lung cancer', image_type=['x_ray', 'ct'])  # Results Found: 9,220.

    # 4. Pull the data
    search_df = opi.pull()

Cancer Imaging Archive
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # 1. Import the interface for the Cancer Imaging Archive
    from biovida.images import CancerImageInterface

    # 2. Create an Instance of the Tool
    cii = CancerImageInterface(YOUR_API_KEY_HERE)

    # 3. Perform a search
    cii.search(cancer_type='esophageal')

    # 4. Pull the data
    cdf = cii.pull()

Both ``CancerImageInterface`` and ``OpeniInterface`` cache images for
later use. When data is 'pulled', a ``records_db`` is generated, which
is a dataframe of all text data associated with the images. They are
provided as class attributes, e.g., ``cii.records_db``. While
``records_db`` only stores data from the most recent data pull,
``cache_records_db`` dataframes provides an account of all image data
currently cached.

Splitting Images
^^^^^^^^^^^^^^^^

BioVida can divide cached images into train/validation/test.

.. code:: python

    from biovida.images import image_divvy

    # 1. Define a rule to 'divvy' up images in the cache.
    def my_divvy_rule(row):
        if row['image_modality_major'] == 'x_ray':
            return 'x_ray'
        elif row['image_modality_major'] == 'ct':
            return 'ct'

    # 2. Define Proportions and Divide Data
    tt = image_divvy(opi, my_divvy_rule, action='ndarray', train_val_test_dict={'train': 0.8, 'test': 0.2})

    # 3. The resultant ndarrays can be unpacked as follows:
    train_ct, train_xray = tt['train']['ct'], tt['train']['x_ray']
    test_ct, test_xray = tt['test']['ct'], tt['test']['x_ray']

Images: Experimental
--------------------

Automated Image Data Cleaning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unfortunately, the data pulled from Open-i above is likely to contain a
large number of images unrelated to the search query and/or are
unsuitable for machine learning.

The *experimental* ``OpeniImageProcessing`` class can be used to
completely automate this data cleaning process, which is partly powered
by a Convolutional Neural Network.

.. code:: python

    # 1. Import Image Processing Tools
    from biovida.images import OpeniImageProcessing

    # 2. Instantiate the Tool using the OpeniInterface Instance
    ip = OpeniImageProcessing(opi)
     
    # 3. Analyze the Images
    idf = ip.auto()

    # 4. Use the Analysis to Clean Images
    ip.clean_image_dataframe()

It is easy to split these images into training and test sets.

.. code:: python

    from biovida.images import image_divvy

    def my_divvy_rule(row):
        if row['image_modality_major'] == 'x_ray':
            return 'x_ray'
        elif row['image_modality_major'] == 'ct':
            return 'ct'

    tt = image_divvy(ip, my_divvy_rule, action='ndarray', train_val_test_dict={'train': 0.8, 'test': 0.2})
    # These ndarrays can be unpack as shown above.

Genomic Data
------------

While primarily focused on images, BioVida also provides a simple
interface for obtaining related information, such genomic data.

.. code:: python

    # 1. Import the Interface for DisGeNET.org
    from biovida.genomics import DisgenetInterface

    # 2. Create an Instance of the Tool
    dna = DisgenetInterface()

    # 3. Pull a Database
    gdf = dna.pull('curated')

Diagnostic Data
---------------

BioVida also makes it easy to obtain diagnostics data.

Information on disease definitions, families and synonyms:

.. code:: python

    # 1. Import the Interface for DiseaseOntology.org
    from biovida.diagnostics import DiseaseOntInterface

    # 2. Create an Instance of the Tool
    doi = DiseaseOntInterface()

    # 3. Pull the Database
    ddf = doi.pull()

Information on symptoms associated with diseases:

.. code:: python

    # 1. Import the Interface for Disease-Symptoms Information
    from biovida.diagnostics import DiseaseSymptomsInterface

    # 2. Create an Instance of the Tool
    dsi = DiseaseSymptomsInterface()

    # 3. Pull the Database
    dsdf = dsi.pull()

Unifying Information
--------------------

The ``unify_against_images`` function integrates image data information
against ``DisgenetInterface``, ``DiseaseOntInterface`` and
``DiseaseSymptomsInterface``.

.. code:: python

    from biovida.unification import unify_against_images

    unify_against_images(interfaces=[cii, opi], db_to_extract='cache_records_db')

Left side of DataFrame: Image Data Alone

+----+----------+--------+-----------+------------------------+-----+-------+------------+-----+
|    | article\ | image\ | image\_ca | modality\_best\_guess  | age | sex   | disease    | ... |
|    | _type    | _id    | ption     |                        |     |       |            |     |
+====+==========+========+===========+========================+=====+=======+============+=====+
| 0  | case\_re | 1      | ...       | Magnetic Resonance     | 73  | male  | fibroma    | ... |
|    | port     |        |           | Imaging (MRI)          |     |       |            |     |
+----+----------+--------+-----------+------------------------+-----+-------+------------+-----+
| 1  | case\_re | 2      | ...       | Magnetic Resonance     | 73  | male  | fibroma    | ... |
|    | port     |        |           | Imaging (MRI)          |     |       |            |     |
+----+----------+--------+-----------+------------------------+-----+-------+------------+-----+
| 2  | case\_re | 1      | ...       | Computed Tomography    | 45  | femal | bile duct  | ... |
|    | port     |        |           | (CT): angiography      |     | e     | cancer     |     |
+----+----------+--------+-----------+------------------------+-----+-------+------------+-----+

Right side of DataFrame: Added Information

+----------------+-------------+------------+---------------+------------+--------------+
| disease\_famil | disease\_sy | disease\_d | known\_associ | mentioned\ | known\_assoc |
| y              | nonym       | efinition  | ated\_symptom | _symptoms  | iated\_genes |
|                |             |            | s             |            |              |
+================+=============+============+===============+============+==============+
| (cell type     | nan         | nan        | (abdominal    | (pain,)    | ((ANTXR2,    |
| benign         |             |            | pain,...)     |            | 0.12), ...)  |
| neoplasm,)     |             |            |               |            |              |
+----------------+-------------+------------+---------------+------------+--------------+
| (cell type     | nan         | nan        | (abdominal    | (pain,)    | ((ANTXR2,    |
| benign         |             |            | pain,...)     |            | 0.12), ...)  |
| neoplasm,)     |             |            |               |            |              |
+----------------+-------------+------------+---------------+------------+--------------+
| (biliary tract | (bile duct  | A biliary  | (abdominal    | (colic,)   | nan          |
| cancer,)       | tumor,...)  | tract...   | obesity,..)   |            |              |
+----------------+-------------+------------+---------------+------------+--------------+

--------------

Documentation
-------------

-  `Getting
   Started <https://tariqahassan.github.io/BioVida/GettingStarted.html>`__
-  `Tutorials <http://nbviewer.jupyter.org/github/tariqahassan/BioVida/tree/master/tutorials/>`__
-  `API
   Documentation <https://tariqahassan.github.io/BioVida/API.html>`__

Contributing
------------

For more information on how to contribute, see the
`contributing <https://github.com/TariqAHassan/BioVida/blob/master/CONTRIBUTING.md>`__
document.

Bug reports and feature requests are always welcome and can be provided
through the `Issues <https://github.com/TariqAHassan/BioVida/issues>`__
page.

Resources
---------

The
`resources <https://github.com/TariqAHassan/BioVida/blob/master/RESOURCES.md>`__
document provides an account of all data sources and scholarly work used
by BioVida.
