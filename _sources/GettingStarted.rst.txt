Getting Started
---------------

This library is primarily intended to automate the collection, post-processing and integration of biomedical data stored
in public online databases. It is hoped that this effort will catalyze new insights and understanding by transforming
multiple, distinct data repositories into unified datasets which are highly amenable to data analysis.

BioVida aims to curate a broad range of biomedical information. In areas such as diagnostics and genomics, this
involves drawing on the work of others, such as the impressive work by the Disease Ontology and DisGeNET teams.
In the case of image data however, BioVida itself performs the 'heavy lifting' involved in collecting and processing
raw data from sources. This is made possible by combining traditional programmatic solutions with recent advances
in machine learning, namely convolutional neural networks.

The guide below provides a brief introduction to getting started with BioVida.

--------------

Installation
------------

Python Package Index:

.. code:: bash

    $ pip install biovida

Latest Build:

.. code:: bash

    $ pip install git+git://github.com/TariqAHassan/BioVida@master

Note: if you are using python on macOS or linux with Python 3, you may wish to use ``pip3 install`` instead.

--------------

Dependencies
------------

BioVida requires: `beautiful soup <https://www.crummy.com/software/BeautifulSoup/>`__,
`h5py <http://www.h5py.org>`__,
`keras <https://keras.io>`__,
`lxml <https://github.com/lxml/lxml>`__,
`numpy <http://www.numpy.org>`__,
`pandas <http://pandas.pydata.org>`__,
`pillow <https://github.com/python-pillow/Pillow>`__,
`pydicom <https://github.com/darcymason/pydicom>`__,
`requests <http://docs.python-requests.org/en/master/>`__,
`scikit-image <http://scikit-image.org>`__,
`scipy <https://www.scipy.org>`__,
`theano <http://deeplearning.net/software/theano/>`__ and
`tqdm <https://github.com/tqdm/tqdm>`__.

The installer will automatically install all of these packages.

**Notes**:

1. Keras is used to power the convolutional neural networks in this project.

2. To use ``scipy`` on macOS (formerly OSX) you will need ``gcc``, which can be obtained with ``homebrew`` via.
   ``$ brew install gcc``. If you do not have ``homebrew`` installed, it can be installed by following the instructions
   provided `here <https://brew.sh>`__.
