BioVida
=======

###Overview

BioVida is a library designed to automate both the harvesting and 
post-processing of biomedical data. The novel datasets it produces
are intended to need little, if any, cleaning by the user.

To view this project's website, please [click here].

------------------------------------------------------------------------

###Installation

Latest Build:
```bash
$ pip install git+git://github.com/TariqAHassan/BioVida@master
```

------------------------------------------------------------------------

###Dependencies

BioVida requires: [pandas], [numpy], [requests], [tqdm], [scipy] and [keras].

------------------------------------------------------------------------

###Image Harvesting


####Import the Interface for the NIH's Open-i API.
```python
from biovida.images.openi_interface import OpenInterface
```

####Create an Instance of the Tool
```python
opi = OpenInterface()
```

####Perform a Search
```python
opi.search("aneurysm", image_type=['mri', 'pet'])
# Results Found: 1,586.
```

The values accepted by the `image_type` argument above can easily be reviewed:
```python
opi.options(search_parameter='image_type')
```

Additionally, searches can easily be reviewed:
```python
opi.current_search
# {'image_type': ['mri', 'pet', 'ct', 'exclude_graphics'], 'query': 'aneurysm'}

opi.current_search_total
# 1165
```

####Pull the data
```python
df = opi.pull()
```

The DataFrame created above, `df`, contains data from all fields provided by the Open-i API.<sup>†</sup>
Images referenced in the DataFrame will automatically be harvested (unless specified otherwise).

<sup>†</sup>*Note*: by default, data harvesting is truncated after the first 60 results.

------------------------------------------------------------------------

###Genomic Data


####Import the Interface for DisGeNET.
```python
from biovida.genomics.disgenet_interface import DisgenetInterface
```

####Create an Instance of the Tool
```python
dna = DisgenetInterface()
```

####Options: Explore Available Databases
```python
dna.options()
# Available Databases:
#   - 'all'
#   - 'curated'
#   - 'snp_disgenet'

dna.options('curated')
# - Full Name:    Curated Gene-Disease Associations
# - Description:  The file contains gene-disease associations from UNIPROT, CTD (human subset),
#                 ClinVar, Orphanet, and the GWAS Catalog.
```

####Pull the data
```python
df = dna.pull('curated')
```
This database will be cached to allow to fast access in the future.

As with the `OpenInterface()` class above, it is easy to gain access to the most recent `pull` and related information.

The database its self:
```python
dna.current_database
```

Information about the database:
```python
dna.current_database_name
# 'curated'

dna.current_database_full_name
# 'Curated Gene-Disease Associations'

dna.current_database_description
# 'The file contains gene-disease associations from...'
```

------------------------------------------------------------------------

###Documentation

For documentation please click [here].

------------------------------------------------------------------------

###Resources

Images

   - The [Open-i] BioMedical Image Search Engine (NIH)

Genomics

   - [DisGeNET]:

      * Janet Piñero, Àlex Bravo, Núria Queralt-Rosinach, Alba Gutiérrez-Sacristán, Jordi Deu-Pons, Emilio Centeno, 
      Javier García-García, Ferran Sanz, and Laura I. Furlong. DisGeNET: a comprehensive platform integrating 
      information on human disease-associated genes and variants. Nucl. Acids Res. (2016) doi:10.1093/nar/gkw943
      
      * Janet Piñero, Núria Queralt-Rosinach, Àlex Bravo, Jordi Deu-Pons, Anna Bauer-Mehren, Martin Baron, 
      Ferran Sanz, Laura I. Furlong. DisGeNET: a discovery platform for the dynamical exploration of human 
      diseases and their genes. Database (2015) doi:10.1093/database/bav028


------------------------------------------------------------------------

###Outstanding Objectives

   1. Images
   
     - Stabalize the process to automatically clean images to make them amenable to machine learning algorithms.
       Part of this problem will be solved using convolutional neural networks (CNNs) as I've found them to 
       produce the best results with small image datasets. This will be performing using the [Keras] library,
       which allows users to use either TensorFlow or Theano as a computational backend. 
    
   2. Diagnostic Data
   
     - source currently unclear


[click here]: https://tariqahassan.github.io/BioVida/index.html
[Keras]: https://keras.io
[pandas]: http://pandas.pydata.org
[numpy]: http://www.numpy.org
[requests]: http://docs.python-requests.org/en/master/
[tqdm]: https://github.com/tqdm/tqdm
[scipy]: https://www.scipy.org
[Open-i]: https://openi.nlm.nih.gov
[DisGeNET]: http://www.disgenet.org/web/DisGeNET/menu
[here]: https://tariqahassan.github.io/BioVida/API.html






