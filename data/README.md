Convolutional Neural Network Training
=====================================

This directory contains the tools to create the training 
data for the convolutional neural network in the
``biovida.images.models.image_classification`` module.


Image Cleaning
==============

**Modalities Considered: X-Ray, CT and MRI**

Outline for ``biovida.images.models.image_processing``:


|              Problem             |                      Action                      |      Status      | Requires Machine Learning |
|:--------------------------------:|:------------------------------------------------:|:----------------:|:-------------------------:|
|        Check if grayscale        | mark finding in dataframe; ban based on modality |      Solved      |             No            |
|      Look for MedPix(R) logo     |               if true, try to crop               |      Solved      |             No            |
|         Look for text bar        |                 if true, try crop                |      Solved      |             No            |
|          Look for border         |               if true, try to crop               |      Solved      |             No            |
|     Look for arrows or boxes     |                   if true, ban                   | Partially Solved |            Yes            |
|       Look for image grids       |                   if true, ban                   | Partially Solved |            Yes            |
| Look for other text in the image |           if true, ban (or find crop).           | Partially Solved |            Yes            |

