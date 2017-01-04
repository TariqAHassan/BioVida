BioVida
===


###Overview:

Publicly available online repositories currently store enormous amounts of data on numerous human diseases. However, 
these databases are often built to serve wildly different purposes, making it difficult to explore connections between 
them. This project aims to tame this problem and integrate several of these public sources together.
Specifically, it aims to develop an easy-to-use API which will harvest the latest information on human diseases from 
public databases.  

------------------------------------------------------------------------

###Outline of Objectives:

   1. Images
   
     - from the NIH's *Open-i* Database (and perhaps others)
        
     - these images will be automatically processed to make them amenable to machine learning algorithms.

   2. Genomic Data
   
     - likely from the NIH's *Genome* database
    
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


####Image Harvesting


#####Import the Interface for the NIH's Open-i API.
```python
from biovida.images.openi_interface import OpenInterface
```

#####Create an Instance of the Tool
```python
io = OpenInterface()
```

#####Perform a Search
```python
io.search("caudate nucleus", image_type=['mri', 'pet', 'ct'])

# Results Found: 1,165.
```

#####Pull the data from the API
```python
df = io.pull()
```

The DataFrame created above, `df`, contains data from all fields provided by the Open-i API.<sup>†</sup>
Images referenced in the DataFrame will automatically be harvested (unless specified otherwise).

<sup>†</sup>*Note*: by default, data harvesting is truncated after the first 60 results.

------------------------------------------------------------------------

##Resources

Images: the NIH's [Open-i] BioMedical Image Search Engine

[inflect]: https://pypi.python.org/pypi/inflect
[pandas]: http://pandas.pydata.org
[numpy]: http://www.numpy.org
[requests]: http://docs.python-requests.org/en/master/
[tqdm]: https://github.com/tqdm/tqdm
[Open-i]: https://openi.nlm.nih.gov





