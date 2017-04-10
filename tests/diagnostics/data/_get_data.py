"""

    Obtain Test Data for the Diagnoistics Subpackage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import pickle
import requests


# ----------------------------------------------------------------------------------------------------------
# DiseaseOntInterface
# ----------------------------------------------------------------------------------------------------------


disease_ontology_db_url='http://purl.obolibrary.org/obo/doid.obo'
obo_file = requests.get(disease_ontology_db_url, stream=True).text.split("[Typedef]")[0]
pickle.dump(obo_file, open("tests/diagnostics/data/obo_file.p", "wb"))


# ----------------------------------------------------------------------------------------------------------
# DiseaseSymptomsInterface
# ----------------------------------------------------------------------------------------------------------


hsdn_url = "https://raw.githubusercontent.com/LABrueggs/HSDN/master/Combined-Output.tsv"
hsdn_file = requests.get(hsdn_url, stream=True).text
pickle.dump(hsdn_file, open("tests/diagnostics/data/hsdn.p", "wb"))


rephetio_ml_url = "https://github.com/dhimmel/medline/raw/0c9e2905ccf8aae00af5217255826fe46cba3d30/data/disease-symptom-cooccurrence.tsv"
rephetio_ml_file = requests.get(rephetio_ml_url, stream=True).text
pickle.dump(rephetio_ml_file, open("tests/diagnostics/data/rephetio_ml.p", "wb"))
