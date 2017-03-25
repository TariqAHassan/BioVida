Contributing
============


## Code

- Unfortunately, this project is still very experimental,
  thus a uniform coding style has not been imposed. Generally however,
  PEP8 is preferred.
  
- Line length: < 120 characters (sorry PEP8)


## API


- The Cancer Image Archive interface is stable, but the code is far
too complex. It needs to be simplified.

- The Open-i data is *very* hard to clean. Two main outstanding
problems:

   1. Extracting a patient's disease(s) from available text when it (they) are
      not given explicitly. Currently, the first disease encounter is used
      when this occurs -- employing more sophisticated, like the NegEx algorithm
      could be helpful. Expertise in NLP would go along way here.
     
   2. Detecting visual problems in images, such as arrows or text. 
      These images need to be detected so they can be removed from any
      training (/test) set.
      
  The current solution for processing images uses a combination of text analysis,
  non-ML image processing algorithms to crop problems (e.g., fast normalized cross-correlation
  to remove logos) as well as convolutional neural networks (the most challenging).
  
  In short, the task for the CNN is to flag visual problems (which were not mentioned in
  the text associated with the image, like arrows), so these images can be excluded from
  any dataset. The current approach uses data synthesis, though it may be more effective
  to use transfer learning with, say, VGG19 where possible. 
  
  This brings me to a more general problem with the CNN: backend API. I am currently
  using ``Keras`` simply because it allows users to use *either* ``tensorflow`` or ``theano``.
  Switching over to ``tensorflow``, ``theano`` or, my (new) preference, PyTorch, would be nice.
  
  
## Unit Testing

This really needs to be built out.


## Documentation

On last count, this project has ~6000 lines of documentation, with ~5300 lines of actual code.
However, several support modules and large parts of the ``images.openi_interface``
module remain undocumented.


## User Experience

- General suggestions about how to make the API 
  easier to use and more feature-complete are welcome!
