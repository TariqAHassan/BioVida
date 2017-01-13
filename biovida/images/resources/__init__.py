# Imports
import os
import requests
from PIL import Image
from biovida.images.models.temp import resources_path # update this


def _medpix_logo_download(image_web_address='https://openi.nlm.nih.gov/imgs/512/341/1544/MPX1544_synpic54565.png'):
    """

    Download and save the MedPix logo from a representative image.
    (The image is needed for image processing).

    :param image_web_address:
    :type image_web_address: ``str``
    """
    save_path = os.path.join(resources_path, 'medpix_logo.png')

    if not os.path.isfile(save_path):
        # Get representative images
        img = Image.open(requests.get(image_web_address, stream=True).raw)
        # Crop and Save
        img_cropped = img.crop((406, 6, 502, 27))
        img_cropped.save(save_path)
        print("\nMedPix Logo, required for image post-processing, has been saved to:\n'{0}'.".format(save_path))



_medpix_logo_download()
