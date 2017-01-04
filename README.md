BioVida
===


###Overview

Publicly available online repositories currently store enormous amounts of data on numerous human diseases. However, 
these databases are often built to serve wildly different purposes, making it difficult to explore connections between 
them. This project aims to tame this problem and integrate several of these public sources together.
Specifically, it aims to develop an easy-to-use API which will harvest the latest information on human diseases from 
public databases. These capabilties will be complimented by other tools to automate the post-processing of the harvest 
data to make machine learning tractable.

------------------------------------------------------------------------

###Outline of Objectives

   1. Images
   
     - from the NIH's [Open-i] database (and perhaps others)
        
     - these images will be automatically processed (likely using neural nets) to make them amenable to machine learning algorithms.

   2. Genomic Data
   
     - from the [DisGeNET] database
    
   3. Diagnostic Data
   
     - symptom information
        
     - source currently unclear

------------------------------------------------------------------------

###Installation

Latest Build:
```bash
$ pip install git+git://github.com/TariqAHassan/BioVida@master
```

------------------------------------------------------------------------

###Dependencies

BioVida requires: [inflect], [pandas], [numpy], [requests] and [tqdm].

------------------------------------------------------------------------

###Examples


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
opi.search("caudate nucleus", image_type=['mri', 'pet', 'ct'])

# Results Found: 1,165.
```

The values accepted by the `image_type` argument above can easily be reviewed:
```python
opi.options(search_parameter='image_type')

# - 'ct'
# - 'graphic'
# - 'mri'
# - 'microscopy'
# - 'pet'
# - 'photograph'
# - 'ultrasound'
# - 'x-ray'
```

Additionally, searches can easily be reviewed:
```python
opi.current_search

# {'image_type': ['mri', 'pet', 'ct', 'exclude_graphics'], 'query': 'caudate nucleus'}

opi.current_search_total

# 1165
```

####Pull the data
```python
df = opi.pull()
```

The DataFrame created above, `df`, contains data from all fields provided by the Open-i API.<sup>†</sup>
Images referenced in the DataFrame will automatically be harvested (unless specified otherwise).


The most recent dataframe obtained by `OpenInterface().pull()` is also saved as an attribute of the class instance.
This dataframe be accessed as follows:
```python
opi.current_search_dataframe
```

<sup>†</sup>*Note*: by default, data harvesting is truncated after the first 60 results.



###Genomic Data from DisGeNET


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


As with the `OpenInterface()` above, it is easy to gain access to the most recent `pull` and related information.

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

###Resources

Images:

   - The [Open-i] BioMedical Image Search Engine (NIH)

Genomics:

   - [DisGeNET]:

      * Janet Piñero, Àlex Bravo, Núria Queralt-Rosinach, Alba Gutiérrez-Sacristán, Jordi Deu-Pons, Emilio Centeno, 
      Javier García-García, Ferran Sanz, and Laura I. Furlong. DisGeNET: a comprehensive platform integrating 
      information on human disease-associated genes and variants. Nucl. Acids Res. (2016) doi:10.1093/nar/gkw943
      
      * Janet Piñero, Núria Queralt-Rosinach, Àlex Bravo, Jordi Deu-Pons, Anna Bauer-Mehren, Martin Baron, 
      Ferran Sanz, Laura I. Furlong. DisGeNET: a discovery platform for the dynamical exploration of human 
      diseases and their genes. Database (2015) doi:10.1093/database/bav028


[inflect]: https://pypi.python.org/pypi/inflect
[pandas]: http://pandas.pydata.org
[numpy]: http://www.numpy.org
[requests]: http://docs.python-requests.org/en/master/
[tqdm]: https://github.com/tqdm/tqdm
[Open-i]: https://openi.nlm.nih.gov
[DisGeNET]: http://www.disgenet.org/web/DisGeNET/menu







