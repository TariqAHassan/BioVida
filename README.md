<div align="center">
  <img src="https://github.com/TariqAHassan/BioVida/blob/master/docs/logo/biovida_logo_regular_scaled.png"><br>
</div>

-----------------

BioVida is a library designed to automate the harvesting, 
post-processing and integration of biomedical data. This automation
is accomplished with a combination of traditional software development approaches
and the latest machine learning techniques, namely convolutional
neural networks.

To view this project's website, please [click here].

## Installation

Latest Build:
```bash
$ pip install git+git://github.com/TariqAHassan/BioVida@master
```

## API Overview

### Image Data

In just a few lines of code, you can download and clean images from various biomedical image databases.

#### Cancer Imaging Archive
```python
# 1. Import the interface for the Cancer Imaging Archive
from biovida.images import CancerImageInterface

# 2. Create an Instance of the Tool
cii = CancerImageInterface(YOUR_API_KEY_HERE)

# 3. Perform a search
cii.search(location='extremities')

# 4. Pull the data
cdf = cii.pull()
```

#### Open-i BioMedical Image Search Engine
```python

# 1. Import the Interface for the NIH's Open-i API.
from biovida.images import OpeniInterface

# 2. Create an Instance of the Tool
opi = OpeniInterface()
 
# 3. Perform a general search for MRIs and CTs
opi.search(query=None, image_type=['mri', 'ct'])  # Results Found: 134,113.

# 4. Pull the data
search_df = opi.pull()
```

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

Notes:
 
   1. This library is in *pre-alpha*. That is, formal unit testing has
      not yet been implemented. **Until it is, this software should be 
      considered to be experimental**.
   
   2. The model has been trained and validated to detect two kinds of problems:
      arrows in the image and 'grids' of images. This was performed using
      a total of 99,000 images, synthesized from a collection of ~1,500 CT
      and *structural* MRI scans.*
      
   3. While the model will likely generalize to x-ray and ultrasound images,
      this has not been formally tested. In the future, the model will be 
      explicitly trained on these types of images.
   
   4. *For images which are not grayscale, such a photographs, fMRI and PET scans,
      the model is almost certain to provide completely erroneous predictions*.
   
*While this may raise concerns about overfitting, it is important to note
that the model was tasked with differentiating between images which had been permuted 
(e.g., had arrows added) with those that had not (random cropping notwithstanding). 
Moreover, in informal testing, this model appears to be be performing very well with new data.

### Genomic Data

BioVida provides a simple interface for obtaining genomic data.

```python
# 1. Import the Interface for DisGeNET.org
from biovida.genomics import DisgenetInterface

# 2. Create an Instance of the Tool
dna = DisgenetInterface()

# 3. Pull a Database
gdf = dna.pull('curated')
```

### Diagnostic Data

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

## Documentation

You can view a more extensive Getting Started guide by [clicking here]
and API documentation [here].

## Resources

The [resources] document provides an account of all data sources and
scholarly work used by BioVida.
   
   
[click here]: https://tariqahassan.github.io/BioVida/index.html
[clicking here]: https://tariqahassan.github.io/BioVida/GettingStarted.html
[here]: https://tariqahassan.github.io/BioVida/API.html
[resources]: https://github.com/TariqAHassan/BioVida/blob/master/RESOURCES.md

