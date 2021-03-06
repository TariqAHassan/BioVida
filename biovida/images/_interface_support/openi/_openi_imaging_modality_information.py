# coding: utf-8

"""

    Information for ``openi_text_processing._imaging_modality_guess()``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from copy import deepcopy
from itertools import combinations

from biovida.support_tools.support_tools import cln

# Note: all terms for matching must be lower case (_imaging_modality_guess() lowers all strings).


# ----------------------------------------------------------------------------------------------------------
# Terms Dict
# ----------------------------------------------------------------------------------------------------------


terms_dict = {
    # format: {abbreviated name, ([alternative names], formal name)}
    # Note: formal names are intended to be collinear with `biovida.images._resources.openi_parameters`'s
    # `openi_image_type_modality_full` dictionary.
    "ct": (['ct ', 'ct:', 'ct-', ' ct', ' ct ', '(ct)',
            'computed tomography', '(ct)'], 'Computed Tomography (CT)'),
    "mri": (['mr ', ' mr ', 'mri ', 'mri:', 'mri-', ' mri', ' mri ', '(mri)',
             'magnetic resonance imag'], 'Magnetic Resonance Imaging (MRI)'),
# 'imag' catches 'es' and 'ing'.
    "pet": (['pet:', 'pet-', ' pet ', '(pet)',
             'positron emission tomography'], 'Positron Emission Tomography (PET)'),
    "photograph": (['photograph'], 'Photograph'),
    "ultrasound": (['ultrasound'], 'Ultrasound'),
    "mammogram": (['mammogram', 'mammograph'], 'Mammography'),
    "x-ray": (['xray', 'x-ray'], 'X-Ray')
}

# ----------------------------------------------------------------------------------------------------------
# Modality Specific Subtypes
# ----------------------------------------------------------------------------------------------------------


# Define a dictionary to use when the modality (term) itself is not present
# e.g., t2 is but MRI is not.
# Note: the first item in the lists below are the 'flag bearers'.
modality_specific_subtypes = {
    'ct': [
        ['angiography', 'angiogram', 'angio-ct'],
        [' mpi '],
        ['segmentation'],
    ],
    'mri': [
        [' gadolinium ', ' gad '],
        ['post-gadolinium', 'post-gad', 'post gad ', ' post gad'],
        [' t1 ', 't1 ', 't1-', 't1w'],
        [' t2 ', 't2 ', 't2-', 't2w'],
        [' stir ', 'stir ', ' stir'],
        [' spgr ', 'spgr ', ' spgr'],
        [' fse ', 'fse ', ' fse '],
        ['flair'],
        [' mra ', 'mra ', ' mra'],
        [' dwi ', 'diffusion weighted'],
        [' dti ', 'diffusion tensor']
    ],
}

# ----------------------------------------------------------------------------------------------------------
# Modality Subtypes. Not specific & cannot be used to infer the modality (like 'T1', for example).
# ----------------------------------------------------------------------------------------------------------


# Divvy up terms which can be used if a match for ``terms_dict`` is successful,
# i.e., if the modality is known, these terms can be used to increase specificity.
contrast = ['contrast-enhanced', 'contrast enhanced', 'enhanced contrast', 'w/contrast',
            'with contrast']

ct_mri_xray = [
    ['abdomen', 'abdominal'],
    ['chest', 'lung']
]

ct_and_mri = [
    ['brain', 'cranial', 'cranium'],
    ['spinal', 'spine'],
    ['non-contrast', 'non contrast', 'noncontrast', 'w/o contrast',
     'without contrast', 'unenhanced'],
    ['pre-contrast', 'precontrast'],
    ['post-contrast', 'postcontrast'],
]

# Define a dictionary to use when there is a match for an item in ``terms_dict``.
modality_subtypes = deepcopy(modality_specific_subtypes)
modality_subtypes['ct'] += ct_mri_xray + ct_and_mri + [
    contrast + [' contrast ct ', ' contrast-ct ']]
modality_subtypes['mri'] += ct_mri_xray + ct_and_mri + [
    contrast + [' contrast mr ', ' contrast-mr ']]
modality_subtypes['x-ray'] = ct_mri_xray

# ----------------------------------------------------------------------------------------------------------
# Contradictions
# ----------------------------------------------------------------------------------------------------------


# Add contradictions to block
contradictions = [
    ['contrast-enhanced', 'non-contrast'],
    ['pre-contrast', 'post-contrast'],
    ['gadolinium', 'post-gadolinium'],
]

anatomical = ['abdomen', 'brain', 'chest', 'spinal']
scan_types = [cln(m[0]) for m in modality_specific_subtypes['mri']]

for i in (anatomical, scan_types):
    contradictions += list(map(list, combinations(i, r=2)))
