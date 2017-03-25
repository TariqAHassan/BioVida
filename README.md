<div align="center">
  <img src="https://github.com/TariqAHassan/BioVida/blob/master/docs/logo/biovida_logo_regular_scaled.png"><br>
</div>

----

BioVida is a experimental library designed to automate the harvesting, 
post-processing and integration of biomedical data.

It is hoped that by freeing machine learning experts from these onerous
tasks, they will be able to focus their efforts on modeling itself. In turn, 
enabling them to advance new insights and understandings into disease.

In a nod to recursion, this library tries to accomplish some of this automation
using machine learning itself (namely, as convolutional neural networks) to
automatically clean messy data.

To view this project's website, please [click here].


## Installation

Latest Build:
```bash
$ pip install git+git://github.com/TariqAHassan/BioVida@master
```

## Stable API Features

In just a few lines of code, you can download and clean images from biomedical image databases.

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

#### Open-i BioMedical Image Search Engine
```python

# 1. Import the Interface for the NIH's Open-i API.
from biovida.images import OpeniInterface

# 2. Create an Instance of the Tool
opi = OpeniInterface()

# 3. Perform a general search for x-rays and cts of lung cancer
opi.search(query='lung cancer', image_type=['x_ray', 'ct'])  # Results Found: 9,220.

# 4. Pull the data
search_df = opi.pull()
```

Both ``CancerImageInterface`` and ``OpeniInterface`` cache images for later use.
The 'record' of the most recent ``'search'`` and ``'pull'`` is provided
by ``records_db`` dataframes, e.g., ``CancerImageInterface.records_db``.
These dataframe contain dozens of columns of information about the image,
such as the age of the subject. Similarly, ``cache_records_db`` is a dataframe of
*all* images currently cached.


#### Splitting Images

BioVida can divide cached images into train/validation/test.

```python
from biovida.images import image_divvy

# 1. Define a rule to 'divvy' up images in the cache.
def my_divvy_rule(row):
    if isinstance(row['image_modality_major'], str):
        if 'x_ray' == row['image_modality_major']:
            return 'x_ray'
        elif 'ct' == row['image_modality_major']:
            return 'ct'

# 2. Define Proportions and Divide Data
tt = image_divvy(opi, my_divvy_rule, action='ndarray', train_val_test_dict={'train': 0.8, 'test': 0.2})

# 3. The resultant ndarrays can be unpacked as follows:
train_ct, train_xray = tt['train']['ct'], tt['train']['x_ray']
test_ct, test_xray = tt['test']['ct'], tt['test']['x_ray']
```

## Experimental Image Features

#### Automated Image Data Cleaning
```python
# 1. Import Image Processing Tools
from biovida.images import ImageProcessing

# 2. Instantiate the Tool using the OpeniInterface Instance
ip = ImageProcessing(opi)
 
# 3. Clean the Image Data
idf = ip.auto()

# 4. Save the Cleaned Images
ip.save("/save/directory")
```

## Genomic Data

BioVida provides a simple interface for obtaining genomic data.

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

The ``unify_against_images`` function integrates image data information from ``DisgenetInterface``,
``DiseaseOntInterface`` and ``DiseaseSymptomsInterface`` together.

```python
from biovida.support_tools import pandas_pprint
from biovida.unification import unify_against_images

udf = unify_against_images(interfaces=[ip, opi], db_to_extract='cache_records_db')

pandas_pprint(udf, full_cols=True)
```

Left side of DataFrame: Image Data Alone

|   | article_type | image_id | image_caption |          modality_best_guess          | age |   sex  |      disease     | ... |
|:-:|:------------:|:--------:|:-------------:|:-------------------------------------:|:---:|:------:|:----------------:|:---:|
| 0 |  case_report |     1    |      ...      |    Magnetic Resonance Imaging (MRI)   |  73 |  male  |      fibroma     | ... |
| 1 |  case_report |     2    |      ...      |   Computed Tomography (CT): abdomen   |  73 |  male  |      fibroma     | ... |
| 2 |  case_report |     1    |      ...      | Computed Tomography (CT): angiography |  45 | female | bile duct cancer | ... |


Right side of DataFrame: External Information


|        disease_family        |    disease_synonym    |       disease_definition       |        known_associated_symptoms       | mentioned_symptoms | known_associated_genes |
|:----------------------------:|:---------------------:|:------------------------------:|:--------------------------------------:|:------------------:|:----------------------:|
| (cell type benign neoplasm,) |          nan          |               nan              |  (abdominal pain, abnormal reflex,...) |       (pain,)      |  (ANTXR2, 0.12), ...)  |
| (cell type benign neoplasm,) |          nan          |               nan              |  (abdominal pain, abnormal reflex,...) |       (pain,)      |  (ANTXR2, 0.12), ...)  |
|    (biliary tract cancer,)   | (bile duct tumor,...) | A biliary tract cancer that... | (abdominal obesity, abdominal pain,..) |      (colic,)      |           nan          |

---

## Documentation

You can view a more extensive Getting Started guide by [clicking here]
and API documentation [here].


## Contributing

Contributions are always welcome. For more information, see the [contributing] information 
for a somewhat protracted outline of current problems.


## Resources

The [resources] document provides an account of all data sources and
scholarly work used by BioVida.
   
   
[click here]: https://tariqahassan.github.io/BioVida/index.html
[clicking here]: https://tariqahassan.github.io/BioVida/GettingStarted.html
[here]: https://tariqahassan.github.io/BioVida/API.html
[Contributing]: https://github.com/TariqAHassan/BioVida/tree/master/docs/contributing
[resources]: https://github.com/TariqAHassan/BioVida/blob/master/RESOURCES.md
