"""

    Convnet Training Images Directly from Open-i
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Imports
from tqdm import tqdm
from biovida.images import image_divvy
from biovida.images import OpeniInterface
from biovida.images import ImageProcessing

from data.source_data._private.my_file_paths import target_dir

# Start tqdm
tqdm.pandas(desc='status')


# ----------------------------------------------------------------------------------------------------------
# Harvesting
# ----------------------------------------------------------------------------------------------------------

# Note: this entails downloading over 250,000 images and will therefore
# take a fair amount of time to complete.

opi = OpeniInterface()
# opi.search(image_type=['mri', 'ct', 'ultrasound', 'x_ray'])
# df = opi.pull(download_limit=None, image_size=None, clinical_cases_only=False)






















