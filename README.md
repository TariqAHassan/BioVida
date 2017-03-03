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

------------------------------------------------------------------------

###Installation

Latest Build:
```bash
$ pip install git+git://github.com/TariqAHassan/BioVida@master
```

------------------------------------------------------------------------

###Dependencies

BioVida requires: [pandas], [numpy], [requests], [tqdm], [pillow], [pydicom], [h5py], [scipy], [scikit-image] and [keras].

------------------------------------------------------------------------

###API Overview

###Image Data

In just a few lines of code, you can download and clean images from various biomedical image databases.

####Cancer Imaging Archive
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

####Open-i BioMedical Image Search Engine
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

####Automated Image Data Cleaning
```python
# 1. Import Image Processing Tools
from biovida.images import ImageProcessing

# 2. Instantiate the Tool using the OpeniInterface Instance
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
from biovida.genomics import DisgenetInterface

# 2. Create an Instance of the Tool
dna = DisgenetInterface()

# 3. Pull a Database
gdf = dna.pull('curated')
```

------------------------------------------------------------------------

###Diagnostic Data

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

------------------------------------------------------------------------

###Documentation

You can view a more extensive Getting Started guide by [clicking here]
and API documentation [here].

------------------------------------------------------------------------

###Resources

Images

   - The [Open-i] BioMedical Image Search Engine (NIH)
   
   - The [Cancer Imaging Archive]

Genomics

   - [DisGeNET]

      * Janet Piñero, Àlex Bravo, Núria Queralt-Rosinach, Alba Gutiérrez-Sacristán, Jordi Deu-Pons, Emilio Centeno, 
      Javier García-García, Ferran Sanz, and Laura I. Furlong. DisGeNET: a comprehensive platform integrating 
      information on human disease-associated genes and variants. Nucl. Acids Res. (2016). doi:[10.1093/nar/gkw943].
      
      * Janet Piñero, Núria Queralt-Rosinach, Àlex Bravo, Jordi Deu-Pons, Anna Bauer-Mehren, Martin Baron, 
      Ferran Sanz, Laura I. Furlong. DisGeNET: a discovery platform for the dynamical exploration of human 
      diseases and their genes. Database (2015). doi:[10.1093/database/bav028].

Diagnostics

   - [DiseaseOntology]
   
   - Rephetio - Medline
   
      * Daniel Himmelstein, Antoine Lizee, Chrissy Hessler, Leo Brueggeman, Sabrina Chen, Dexter Hadley, Ari Green,
        Pouya Khankhanian, Sergio Baranzini (2016) Rephetio: Repurposing drugs on a hetnet [report].
        Thinklab. doi:[10.15363/thinklab.a7]. Code & Data Repositiory: https://github.com/dhimmel/medline.
        
   - Human Symptoms Disease Network
   
      * Zhou, X., Menche, J., Barabási, A. L., & Sharma, A. (2014). Human symptoms–disease network.
        Nature communications, 5. doi:[10.1038/ncomms5212]. Code & Data Repositiory: https://github.com/dhimmel/hsdn.
   
   
   
[click here]: https://tariqahassan.github.io/BioVida/index.html
[pandas]: http://pandas.pydata.org
[numpy]: http://www.numpy.org
[requests]: http://docs.python-requests.org/en/master/
[tqdm]: https://github.com/tqdm/tqdm
[pillow]: https://github.com/python-pillow/Pillow
[pydicom]: https://github.com/darcymason/pydicom
[h5py]: http://www.h5py.org
[scipy]: https://www.scipy.org
[scikit-image]: http://scikit-image.org
[keras]: https://keras.io
[Open-i]: https://openi.nlm.nih.gov
[DisGeNET]: http://www.disgenet.org/web/DisGeNET/menu
[clicking here]: https://tariqahassan.github.io/BioVida/GettingStarted.html
[here]: https://tariqahassan.github.io/BioVida/API.html
[DiseaseOntology]: http://disease-ontology.org
[Cancer Imaging Archive]: http://www.cancerimagingarchive.net
[10.1093/nar/gkw943]: https://doi.org/10.1093/nar/gkw943
[10.1093/database/bav028]: https://doi.org/10.1093/database/bav028
[10.15363/thinklab.a7]: http://www.thinklab.com/p/rephetio/report
[10.1038/ncomms5212]: http://www.nature.com/articles/ncomms5212
