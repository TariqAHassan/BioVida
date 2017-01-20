<div align="center">
  <img src="https://github.com/TariqAHassan/BioVida/blob/5e96d3cd725a6bb1ad958b9cf46100ea0eead606/docs/logo/biovida_logo_scaled.png"><br>
</div>


BioVida is a library designed to automate both the harvesting and 
post-processing of biomedical data. The novel datasets it produces
are intended to need little, if any, cleaning by the user.
Part of this automation is powered by convolution neural networks
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

BioVida requires: [pandas], [numpy], [requests], [tqdm], [scipy], [scikit-image] and [keras].

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
opi.search("aneurysm", image_type=['mri', 'pet'])  # Results Found: 1,586.

# 4. Pull the data
search_df = opi.pull()
```

####Cleaning Images
```python
# 1. Import Image Processing Tools
from biovida.images.image_processing import ImageProcessing

# 2. Create an Instance of the Tool using the Search Data
ip = ImageProcessing(search_df)
 
# 3. Clean the Image Data
cdf = ip.auto()

# 4. Save the Cleaned Images
ip.save("/save/directory/")
```

Note: this library is still in *pre-alpha*.
Therefore, the procedures that power the above code snippet are
still being refined and improved.

------------------------------------------------------------------------

###Genomic Data

BioVida also provides an easy interface for obtaining
Genomic data.

```python
# 1. Import the Interface for DisGeNET
from biovida.genomics.disgenet_interface import DisgenetInterface

# 2. Create an Instance of the Tool
dna = DisgenetInterface()

# 3. Pull a Database
df = dna.pull('curated')
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


------------------------------------------------------------------------

###Outstanding Objectives

   1. Images
   
     - Stabalize image processing procedures.
    
   2. Diagnostic Data
   
     - likely [disease-ontology.org]


[click here]: https://tariqahassan.github.io/BioVida/GettingStarted.html
[Keras]: https://keras.io
[pandas]: http://pandas.pydata.org
[numpy]: http://www.numpy.org
[requests]: http://docs.python-requests.org/en/master/
[tqdm]: https://github.com/tqdm/tqdm
[scipy]: https://www.scipy.org
[scikit-image]: http://scikit-image.org
[Open-i]: https://openi.nlm.nih.gov
[DisGeNET]: http://www.disgenet.org/web/DisGeNET/menu
[clicking here]: https://tariqahassan.github.io/BioVida/GettingStarted.html
[here]: https://tariqahassan.github.io/BioVida/API.html
[disease-ontology.org]: http://disease-ontology.org

