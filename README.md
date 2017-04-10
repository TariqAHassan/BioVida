<div align="center">
  <img src="https://github.com/TariqAHassan/BioVida/blob/master/docs/logo/biovida_logo_regular_scaled.png"><br>
</div>

----
<p align="center">
  <a href="https://travis-ci.org/TariqAHassan/BioVida">
    <img src="https://api.travis-ci.org/TariqAHassan/BioVida.svg?branch=master"
         alt="build status">
  </a>
</p>

BioVida is a library designed to make it easy to gain access to 
existing data sets of biomedical images as well as build brand new, custom-made ones.

It is hoped that by automating the tedious data munging that is typically involved
in this process, more people will become interested in applying machine learning
to biomedical images and, in turn, advancing insights into human disease.

In a nod to recursion, BioVida tries to accomplish some of this automation
with machine learning itself, using tools like convolutional neural networks.

## Installation

Python Package Index:
```bash
$ pip install biovida
```

Latest Build:
```bash
$ pip install git+git://github.com/TariqAHassan/BioVida@master
```

Requires Python 3.4+

## Images: Stable

In just a few lines of code, you can gain access to biomedical databases
which store tens of millions of images. 

*Please note that you are bound to adhere to the copyright and other 
usage restrictions under which this data is provided to you by its creators.*


#### Open-i BioMedical Image Search Engine
```python

# 1. Import the Interface for the NIH's Open-i API.
from biovida.images import OpeniInterface

# 2. Create an Instance of the Tool
opi = OpeniInterface()

# 3. Perform a search for x-rays and cts of lung cancer
opi.search(query='lung cancer', image_type=['x_ray', 'ct'])  # Results Found: 9,220.

# 4. Pull the data
search_df = opi.pull()
```

#### Cancer Imaging Archive
```python
# 1. Import the interface for the Cancer Imaging Archive
from biovida.images import CancerImageInterface

# 2. Create an Instance of the Tool
cii = CancerImageInterface(YOUR_API_KEY_HERE)

# 3. Perform a search
cii.search(cancer_type='esophageal')

# 4. Pull the data
cdf = cii.pull()
```

Both ``CancerImageInterface`` and ``OpeniInterface`` cache images for later use.
When data is 'pulled', a ``records_db`` is generated, which is a dataframe
of all text data associated with the images. They are provided as class attributes, e.g.,
 ``cii.records_db``. While ``records_db`` only stores data from the most recent data pull,
``cache_records_db`` dataframes provides an account of all image data currently cached.

#### Splitting Images

BioVida can divide cached images into train/validation/test.

```python
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
```

## Images: Experimental

#### Automated Image Data Cleaning

Unfortunately, the data pulled from Open-i above
is likely to contain a large number of images 
unrelated to the search query and/or are unsuitable for machine learning.

The *experimental* ``OpeniImageProcessing`` class can be used to completely
automate this data cleaning process, which is partly powered by a Convolutional Neural Network.

```python
# 1. Import Image Processing Tools
from biovida.images import OpeniImageProcessing

# 2. Instantiate the Tool using the OpeniInterface Instance
ip = OpeniImageProcessing(opi)
 
# 3. Analyze the Images
idf = ip.auto()

# 4. Use the Analysis to Clean Images
ip.clean_image_dataframe()
```

It is easy to split these images into training and test sets.

```python
from biovida.images import image_divvy

def my_divvy_rule(row):
    if row['image_modality_major'] == 'x_ray':
        return 'x_ray'
    elif row['image_modality_major'] == 'ct':
        return 'ct'

tt = image_divvy(ip, my_divvy_rule, action='ndarray', train_val_test_dict={'train': 0.8, 'test': 0.2})
# These ndarrays can be unpack as shown above.
```

## Genomic Data

While primarily focused on images, BioVida also provides a simple interface for
obtaining related information, such genomic data.

```python
# 1. Import the Interface for DisGeNET.org
from biovida.genomics import DisgenetInterface

# 2. Create an Instance of the Tool
dna = DisgenetInterface()

# 3. Pull a Database
gdf = dna.pull('curated')
```

## Diagnostic Data

BioVida also makes it easy to obtain diagnostics data.

Information on disease definitions, families and synonyms: 

```python
# 1. Import the Interface for DiseaseOntology.org
from biovida.diagnostics import DiseaseOntInterface

# 2. Create an Instance of the Tool
doi = DiseaseOntInterface()

# 3. Pull the Database
ddf = doi.pull()
```

Information on symptoms associated with diseases:

```python
# 1. Import the Interface for Disease-Symptoms Information
from biovida.diagnostics import DiseaseSymptomsInterface

# 2. Create an Instance of the Tool
dsi = DiseaseSymptomsInterface()

# 3. Pull the Database
dsdf = dsi.pull()
```

## Unifying Information

The ``unify_against_images`` function integrates image data information against ``DisgenetInterface``,
``DiseaseOntInterface`` and ``DiseaseSymptomsInterface``.

```python
from biovida.unification import unify_against_images

unify_against_images(interfaces=[cii, opi], db_to_extract='cache_records_db')
```

Left side of DataFrame: Image Data Alone

|   | article_type | image_id | image_caption |          modality_best_guess          | age |   sex  |      disease     | ... |
|:-:|:------------:|:--------:|:-------------:|:-------------------------------------:|:---:|:------:|:----------------:|:---:|
| 0 |  case_report |     1    |      ...      |    Magnetic Resonance Imaging (MRI)   |  73 |  male  |      fibroma     | ... |
| 1 |  case_report |     2    |      ...      |    Magnetic Resonance Imaging (MRI)   |  73 |  male  |      fibroma     | ... |
| 2 |  case_report |     1    |      ...      | Computed Tomography (CT): angiography |  45 | female | bile duct cancer | ... |


Right side of DataFrame: Added Information


|        disease_family        |    disease_synonym    | disease_definition | known_associated_symptoms | mentioned_symptoms | known_associated_genes  |
|:----------------------------:|:---------------------:|:------------------:|:-------------------------:|:------------------:|:-----------------------:|
| (cell type benign neoplasm,) |          nan          |        nan         |  (abdominal pain,...)     |       (pain,)      |  ((ANTXR2, 0.12), ...)  |
| (cell type benign neoplasm,) |          nan          |        nan         |  (abdominal pain,...)     |       (pain,)      |  ((ANTXR2, 0.12), ...)  |
|    (biliary tract cancer,)   | (bile duct tumor,...) | A biliary tract... | (abdominal obesity,..)    |      (colic,)      |            nan          |

---

## Documentation

 * [Getting Started]
 * [Tutorials] 
 * [API Documentation]

## Contributing

For more information on how to contribute, see the [contributing] document.

Bug reports and feature requests are always welcome and can be provided through the 
[Issues] page.

## Resources

The [resources] document provides an account of all data sources and
scholarly work used by BioVida.
   
   
[click here]: https://tariqahassan.github.io/BioVida/index.html
[Getting Started]: https://tariqahassan.github.io/BioVida/GettingStarted.html
[Tutorials]: http://nbviewer.jupyter.org/github/tariqahassan/BioVida/tree/master/tutorials/
[API Documentation]: https://tariqahassan.github.io/BioVida/API.html
[Contributing]: https://github.com/TariqAHassan/BioVida/blob/master/CONTRIBUTING.md
[Issues]: https://github.com/TariqAHassan/BioVida/issues
[resources]: https://github.com/TariqAHassan/BioVida/blob/master/RESOURCES.md
