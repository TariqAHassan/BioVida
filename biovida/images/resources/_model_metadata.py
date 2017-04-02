# coding: utf-8

"""

    Metadata on what the Model has Been Trained to Classify
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import json


trained_open_i_modality_types = (
    # Names as they appear in the 'image_modality_major' column and in the
    # ``images._interface_support.openi.openi_parameters.openi_image_type_params``
    # dictionary.
    'ct',
    'mri',
    'x_ray',
)

# Note: this write command is only compatible with Python 3.
with open('./biovida/images/resources/trained_open_i_modality_types.json', 'w') as f:
    json.dump(trained_open_i_modality_types, f, ensure_ascii=False)
