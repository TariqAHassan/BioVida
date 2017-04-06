# coding: utf-8

"""

    General Support Tools for Cancer Image Archive Data Processing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""


# ----------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------


nonessential_cancer_image_columns = [
    # Columns to drop when returning
    # '..._short' dataframes attributes
    # of the CancerImageInterface class.
    'series_number',
    'biovida_version',
    'series_instance_uid',
    'study_instance_uid',
    'series_number',
    'software_versions',
    'visibility',
    'image_count'
]
