"""

    Obtain Test Data for the Diagnoistics Subpackage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import requests


# ----------------------------------------------------------------------------------------------------------
# DiseaseOntInterface - Disease Ontology
# ----------------------------------------------------------------------------------------------------------


disease_ontology_db_url='http://purl.obolibrary.org/obo/doid.obo'
obo_file = requests.get(disease_ontology_db_url, stream=True).text.split("[Typedef]")[0]


with open("tests/diagnostics/data/obo_file.txt", "w") as file:
    file.write(obo_file)


# ----------------------------------------------------------------------------------------------------------
# DiseaseSymptomsInterface - Disease Ontology
# ----------------------------------------------------------------------------------------------------------

































