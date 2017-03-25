Contributing
============


## Code

- Unfortunately, this project is still very experimental,
  thus a uniform coding style has not been imposed. Generally however,
  PEP8 is preferred.
  
- Line length: < 120 characters (sorry PEP8)


## API


**tl;dr**

- The Cancer Image Archive interface is stable, but the code is far
too complex

- The Open-i data is *very* hard to clean. The procedures need to 
be greatly improved: 

   - extracting a patient's disease(s) from available text when it (they) are
     not given explicitly.
     
   - detecting visual problems (neural networks).

           
### The Cancer Image Archive

- The REST API provided by *The Cancer Image Archive* (TCIA) is difficult to 
  encapsulate and abstract for a number of reasons:
  
  - The API does not provide a several pieces of key information,
    such as the diagnosis. A series of elaborate solutions were required
    to obtain complete the provided data.
    
  - The images are provided as DICOM files, a rather exotic 
    file format used exclusively in the medical imaging community.
    These images are processed using the ``pydicom`` library,
    though several studies on TCIA contain images which can not 
    be converted to standard image types (e.g., 'png') because the
    ``ndarray`` that ``pydicom`` generates have more than 3 dimensions.
    Whether ``pydicom`` or the images themselves participate this is not obvious.
    
For these, and many other reasons, the code written to handle 
this source is very verbose and complex.


### Open-i Medical Image Database

- As noted in the *README* in the base of this repository, the
use of Open-i data is highly experimental. The Open-i database
contains over *1.2 million* images. If this could be 'unleashed'
for machine learning would be a great boon. However, there are
several obstacles only some of which have been tackled thus far.
For instance:

   - Only some of the images are complete with the diagnosis 
     made by a trained physician (those from MedPix). Currently
     the diagnosis is guessed using the heuristic that the first
     disease mentioned is usually the patient's diagnosis, however
     this is more than fallible and does not account for 
     comorbidity, for instance. More advanced text processing
     procedures are sorely needed (e.g., the NegEx algorithm).
     
  - Several images contain arrows, text which occludes the image and/or are arrayed as 
    'grids' of images. These problems need to be identified so these images can be
    excluded when generating a dataset. I've tried to do this by:
    
       - Analyzing the text associated with the image (fairly effective)
       
       - Using Convnets. Problems have abounded:
       
            - low amount of problematic images to use for training data can be identified using
              text parsing approaches
            
            - to account for the problem above, additional image data is synthesised. However
              it only roughly similar to the actual data. 
              
            - The accuracy obtained (~0.9) is artificially high because the validation and
              train sets contained far too much synthesised data. In short it's a classic case
              of 'my train set doesn't really match my test set'. 
              
            - The model used, while it has ~5 million params., is too simple to 
              truly do well at this task. It is especially light in the number of
              convolution layers, degrading it ability to evolve representation invariants.
              However, short of spinning up an AWS box, I do not currently have access to 
              a GPU powerful enough to drive something like VGG19.
              
            - While the final solution is somewhat effective, there are a large number of
              false positives and some glaring false negatives.
              
            - Transfer learning is a future direction. There are some arXiv papers where the authors
              have used TL successfully against x-rays, for instance (despite their being
              vastly different than ImageNet images which the models were originally train against).

              
*In short*, the OPEN-i interface is very much a work-in-progress.

  
## Unit Testing

This needs to be built out -- badly.


## Documentation

On last count, this project has ~6000 lines of documentation, with ~5300 lines of actual code.
However, several support modules and large parts of the ``images.openi_interface``
module remain undocumented.


## User Experience

- General suggestions about how to make the API 
  easier to use and more feature-complete are welcome!
