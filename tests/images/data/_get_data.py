"""

    Obtain Test Data for the Images Subpackage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from biovida.images import OpeniInterface
from biovida.images.openi_interface import _OpeniRecords


# ----------------------------------------------------------------------------------------------------------
# OpeniInterface
# ----------------------------------------------------------------------------------------------------------


opi = OpeniInterface(verbose=True)
opi.search()


oir = _OpeniRecords(root_url='https://openi.nlm.nih.gov',
                    date_format='%d/%m/%Y',
                    verbose=True,
                    cache_path='tests/images/data')


db = oir.records_pull(search_url=opi.current_search_url,
                      to_harvest=opi._current_search_to_harvest,
                      total=100,
                      query=opi.current_query,
                      pull_time=opi._pull_time,
                      records_sleep_time=(10, 1.5),
                      clinical_cases_only=False,
                      return_raw=True)


db.to_pickle("tests/images/data/sample_records_raw.p")
