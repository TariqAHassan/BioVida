"""

    Obtain Test Data for the Diagnoistics Subpackage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import requests


# ----------------------------------------------------------------------------------------------------------
# DiseaseOntInterface
# ----------------------------------------------------------------------------------------------------------


disease_ontology_db_url='http://purl.obolibrary.org/obo/doid.obo'
obo_file = requests.get(disease_ontology_db_url, stream=True).text.split("[Typedef]")[0]


with open("tests/diagnostics/data/obo_file.txt", "w") as file:
    file.write(obo_file)


# ----------------------------------------------------------------------------------------------------------
# DiseaseSymptomsInterface
# ----------------------------------------------------------------------------------------------------------


hsdn_url = "https://raw.githubusercontent.com/LABrueggs/HSDN/master/Combined-Output.tsv"
hsdn_file = requests.get(hsdn_url, stream=True).text


with open("tests/diagnostics/data/hsdn.tsv", "w") as file:
    file.write(hsdn_file)


rephetio_ml_url = "https://github.com/dhimmel/medline/raw/0c9e2905ccf8aae00af5217255826fe46cba3d30/data/disease-symptom-cooccurrence.tsv"
rephetio_ml_file = requests.get(rephetio_ml_url, stream=True).text


with open("tests/diagnostics/data/rephetio_ml.tsv", "w") as file:
    file.write(rephetio_ml_file)
