<div align="center">
  <img src="https://github.com/TariqAHassan/BioVida/blob/master/docs/logo/biovida_logo_regular_scaled.png"><br>
</div>

-----------------

BioVida is a library designed to automate both the harvesting and 
post-processing of biomedical data. The novel datasets it produces
are intended to need little, if any, cleaning by the user.
Part of this automation is powered by convolutional neural networks
which operate seamlessly *behind the scenes*, helping to automagically
clean the data.

To view this project's website, please [click here].

------------------------------------------------------------------------

###Installation

Latest Build:
```bash
$ pip install git+git://github.com/TariqAHassan/BioVida@master
```

------------------------------------------------------------------------

###Dependencies

BioVida requires: [pandas], [numpy], [requests], [tqdm], [pillow], [scipy], [scikit-image] and [keras].

------------------------------------------------------------------------

###API Overview

###Image Data

In just a few lines of code, you can download and clean images from
the NIH's [Open-i] Biomedical Search Database.

####Downloading Images
```python

# 1. Import the Interface for the NIH's Open-i API.
from biovida.images.openi_interface import OpenInterface

# 2. Create an Instance of the Tool
opi = OpenInterface()
 
# 3. Perform a search
opi.search("aneurysm", image_type=['mri', 'ct'])  # Results Found: 3,973.

# 4. Pull the data
search_df = opi.pull()
```

####Cleaning Images
```python
# 1. Import Image Processing Tools
from biovida.images.image_processing import ImageProcessing

# 2. Instantiate the Tool using the OpenInterface Instance
ip = ImageProcessing(opi)
 
# 3. Clean the Image Data
idf = ip.auto()

# 4. Save the Cleaned Images
ip.save("/save/directory/")
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

------------------------------------------------------------------------

###Genomic Data

BioVida provides a simple interface for obtaining genomic data.

```python
# 1. Import the Interface for DisGeNET.org
from biovida.genomics.disgenet_interface import DisgenetInterface

# 2. Create an Instance of the Tool
dna = DisgenetInterface()

# 3. Pull a Database
gdf = dna.pull('curated')
```

------------------------------------------------------------------------

###Diagnostic Data

BioVida also makes it easy to obtain diagnostics data.

```python
# 1. Import the Interface for DiseaseOntology.org
from biovida.diagnostics.disease_ont_interface import DiseaseOntInterface

# 2. Create an Instance of the Tool
doi = DiseaseOntInterface()

# 3. Pull the Database
ddf = doi.pull()
```

------------------------------------------------------------------------

###Documentation

You can view a more extensive Getting Started guide by [clicking here]
and API documentation [here].

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

Diagnostics

   - [DiseaseOntology]


[click here]: https://tariqahassan.github.io/BioVida/index.html
[pandas]: http://pandas.pydata.org
[numpy]: http://www.numpy.org
[requests]: http://docs.python-requests.org/en/master/
[tqdm]: https://github.com/tqdm/tqdm
[pillow]: https://github.com/python-pillow/Pillow
[scipy]: https://www.scipy.org
[scikit-image]: http://scikit-image.org
[keras]: https://keras.io
[Open-i]: https://openi.nlm.nih.gov
[DisGeNET]: http://www.disgenet.org/web/DisGeNET/menu
[clicking here]: https://tariqahassan.github.io/BioVida/GettingStarted.html
[here]: https://tariqahassan.github.io/BioVida/API.html
[DiseaseOntology]: http://disease-ontology.org
