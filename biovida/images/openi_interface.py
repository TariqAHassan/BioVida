"""

    Harvest Data from the NIH's Open-i API
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    URL: https://openi.nlm.nih.gov.

    Need to extract Image Type (i.e., mark as: X-Ray, CT, PET, MRI, etc.)

"""
# Imports
import os
import re
import inflect
import requests
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from tqdm import tqdm
from time import sleep
from scipy import misc
from pathlib import Path
from pprint import pprint
from itertools import chain
from datetime import datetime
from easymoney.support_tools import cln
from scipy.ndimage import imread as scipy_imread
from easymoney.easy_pandas import pandas_pretty_print

# Tools used below
non_decimal = re.compile(r'[^\d.]+')
p = inflect.engine()

# To install scipy: brew install gcc; pip3 install Pillow

# ---------------------------------------------------------------------------------------------
# General Terms
# ---------------------------------------------------------------------------------------------

search_query = 'retrieve.php?q=&it=c,m,mc,p,ph,u,x'
age_dict = {p.number_to_words(i): i for i in range(1, 135)}

# ---------------------------------------------------------------------------------------------
# Pulling Data from the NIH's Open-i API
# ---------------------------------------------------------------------------------------------


def openi_bounds_formatter(bounds):
    """Format the computed bounds for the Open-i API."""
    return ["&m={0}&n={1}".format(i[0], i[1]) for i in bounds]

def iter_join(t, join_on="_"):
    return join_on.join(t) if isinstance(t, (list, tuple)) else i

def null_convert(i):
    return None if not i else i

def url_combine(url1, url2):
    return None if any(x is None for x in [url1, url2]) else (url1[:-1] if url1.endswith("/") else url1) + url2

def item_extract(i, list_len=1):
    """Extract the first item in a list or tuple, else return None"""
    return i[0] if isinstance(i, (list, tuple)) and len(i) == list_len else None

def extract_float(i):
    """http://stackoverflow.com/a/947789/4898004"""
    return non_decimal.sub('', i)

def num_word_to_int(input_str):
    """Replace natural numbers from 1 to 130 with intigers."""
    for w, i in age_dict.items():
        for case in [w.upper(), w.lower(), w.title()]: # not perfect, but should do
            input_str = input_str.replace(case, str(i))
    return input_str


def openi_bounds(total, req_limit=30):
    """

    :param total:
    :param req_limit:
    :return:
    """
    # Initalize
    end = 1
    bounds = list()

    # Block invalid values for 'total'.
    if total < 1:
        raise ValueError("'%s' is an invalid value for total" % (total))

    # Block the procedure below if total < req_limit
    if total < req_limit:
        return [(1, total)]

    # Compute the number of steps
    n_steps = total/req_limit

    # Floor the number of steps and loop
    for i in range(int(n_steps)):
        bounds.append((end, end+29))
        end += 30

    # Compute remainder
    remainder = total - (int(n_steps) * req_limit)

    # Add remaining part, if nonzero
    if remainder != 0:
        bounds += [(total - remainder + 1, total)]

    return bounds


def date_format(date_dict, date_format="%d/%m/%Y"):
    """

    :param date_dict:
    :param date_format:
    :return:
    """
    if not date_dict:
        return None

    info = [int(i) if i.isdigit() else None for i in [date_dict['year'], date_dict['month'], date_dict['day']]]
    if info[0] is None or (info[1] is None and info[2] is not None):
        return None

    cleaned_info = [1 if i is None or i < 1 else i for i in info]
    return datetime(cleaned_info[0], cleaned_info[1], cleaned_info[2]).strftime(date_format)


def openi_block_harvest(url, bound, root_url):
    """

    To Harvest:
      - title
      - affiliate
      - articleType
      - authors
      - uid
      - pmcid
      - journal_date (join these).
      - journal_title
      - licenseType
      - MeSH
          -- major
          -- minor
      - Problems
      - abstract
      - link: 'https://openi.nlm.nih.gov/' + imgLarge
      - fulltext_html_url
      - image
          -- captions
          -- id
          -- mention
          -- modalityMajor

    :param url:
    :param bound:
    :param root_url:
    :return:
    """
    # Init
    item_dict = dict()

    # Extract the starting point
    start = int(re.findall('&m=(.+?)&', bound)[0])

    # Define Items to harvest
    to_harvest = ['title', 'affiliate', 'articleType', 'authors', 'uid', 'pmcid',
                  'journal_date', 'journal_title', 'licenseType',
                  ('MeSH', 'major'), ('MeSH', 'minor'), 'Problems',
                  'abstract', 'imgLarge', 'fulltext_html_url', ('image', 'captions'),
                  ('image', 'id'), ('image', 'mention'), ('image', 'modalityMajor')]

    # Request Data from the server
    req = requests.get(url + bound).json()['list']

    # Loop
    list_of_dicts = list()
    for e, item in enumerate(req):
        # Init
        item_dict = {"req_no": start + e}

        # Populate current `item_dict`
        for j in to_harvest:
            if isinstance(j, tuple):
                item_dict[iter_join(j)] = null_convert(item.get(j[0], {}).get(j[1], None))
            elif j == 'journal_date':
                item_dict[j] = null_convert(date_format(item.get(j, None)))
            elif j == 'imgLarge':
                item_dict[j] = url_combine(root_url, null_convert(item.get(j, None)))
            else:
                item_dict[j] = null_convert(item.get(j, None))

        list_of_dicts.append(item_dict)

    return list_of_dicts


def openi_harvest(bounds_list, joined_url, root_url, sleep_mini=(2, 5), sleep_main=(50, 300), verbose=True):
    """

    :param bounds_list:
    :param joined_url:
    :param bound:
    :param root_url:
    :param sleep_mini: (interval, sleep time in seconds)
    :param sleep_main: (interval, sleep time in seconds)
    :return:
    """
    harvested_data = list()
    for e, bound in enumerate(bounds_list):
        if verbose:
            print("Block %s of %s." % (e + 1, len(bounds_list)))
        if e % sleep_mini[0] == 0:
            sleep(abs(sleep_mini[1] + np.random.normal()))
        elif e % sleep_main[0] == 0:
            if verbose:
                print("Sleeping for %s seconds..." % sleep_main[1])
            sleep(abs(sleep_main[1] + np.random.normal()))
        harvested_data += openi_block_harvest(joined_url, bound, root_url)

    # Return
    return harvested_data


def openi_kinesin(search_query, req_limit=30, n_bounds_limit=5, verbose=True):
    """

    'Walk' along the search query and harvest the data.

    :param search_query:
    :param req_limit: Defaults to 30 (limit imposed by Open-i).
    :param n_bounds_limit: temp.
    :param verbose: print information.
    :return:
    """
    root_url = 'https://openi.nlm.nih.gov/'
    joined_url = root_url + search_query

    # Get the total number of results
    total = requests.get(joined_url + "&m=1&n=1").json()['total']

    # Block progress if no results found
    if total < 1:
        raise ValueError("No Results Found.")

    # Print number of results found
    if verbose:
        print("Results Found: %s." % ('{:,.0f}'.format(total)))

    # Compute a list of search ranges to harvest
    bounds_list = openi_bounds_formatter(openi_bounds(total, req_limit=req_limit))

    # Define Extract Limit
    trunc_bounds = bounds_list[0:n_bounds_limit] if isinstance(n_bounds_limit, int) else bounds_list

    # Harvest
    harvested_data = openi_harvest(trunc_bounds, joined_url, root_url)

    # Convert to a DataFrame
    df = pd.DataFrame(harvested_data).fillna(np.NaN)

    # Lower the columns
    df.columns = list(map(lambda x: x.lower(), df.columns))

    # Return
    return df


# ---------------------------------------------------------------------------------------------
# Abstracts Processing
# ---------------------------------------------------------------------------------------------


def mexpix_info_extract(abstract):
    """

    :param abstract:
    :return:
    """
    flags = ['Diagnosis', 'History', 'Findings']
    return {i: item_extract(re.findall('<p><b>' + i + ': </b>(.*?)</p><p>', cln(abstract))) for i in flags}


def patient_sex_guess(abstract):
    """

    Tries to extract the sex of the patient (female or male).

    :param abstract:
    :return:
    """
    counts_dict_f = {t: abstract.lower().count(t) for t in ['female', 'woman', 'girl',' f ']}
    counts_dict_m = {t: abstract.lower().count(t) for t in ['male', 'man', 'boy', ' m ']}

    # Block Conflicting Information
    if sum(counts_dict_f.values()) > 0 and sum([v for k, v in counts_dict_m.items() if k not in ['male', 'man']]) > 0:
        return None

    # Check for sex information
    if any(x > 0 for x in counts_dict_f.values()):
        return 'female'
    elif any(x > 0 for x in counts_dict_m.values()):
        return 'male'
    else:
        return None


def age_refine(age_list):
    """

    :param age_list:
    :return:
    """
    to_return = list()
    for a in age_list:
        age = float(cln(extract_float(a)).strip())

        if 'month' in a:
            to_return.append(round(age / 12, 2))
        else:
            to_return.append(age)

    # Heuristic: typically the largest value will be the age
    return max(to_return)


def patient_age_guess(abstract):
    """

    Forms:
        - x yo
        - x year
        - x-year
        - DOB: x
        - DOB x

    Note: should also consider 'elderly' if no other information can be harvested

    :param abstract:
    :return:
    """
    back = [" y", "yo ", "y.o.", "y/o", "year", "-year", " - year", " -year",
            "month old", " month old", "-month old", "months old", " months old", "-months old"]
    # front = ["dob: ", "dob "]

    # Block: 'x year history'
    history_block = ["year history", " year history", "-year history"]
    hist_matches = [re.findall(r'\d*\.?\d+' + drop, abstract) for drop in history_block]

    # Clean and recompose the string
    cleaned_abstract = cln(" ".join([abstract.replace(r, "") for r in chain(*filter(None, hist_matches))])).strip()
    hist_matches_flat = list(chain(*filter(None, hist_matches)))

    if len(hist_matches_flat):
        cleaned_abstract = num_word_to_int(cln(" ".join([abstract.replace(r, "") for r in hist_matches_flat])).strip())
    else:
        cleaned_abstract = num_word_to_int(cln(abstract))

    # Block processing of empty strings
    if not len(cleaned_abstract):
        return None

    # Try front
    front_finds = list(chain(*filter(None, [re.findall(r'\d+' + b, cleaned_abstract) for b in back])))

    # Return
    return age_refine(front_finds) if len(front_finds) else None


def flag_extract(x):
    """

    To Harvest:
        - Age
        - Sex

    :param abstract:
    :param journal:
    :return:
    """
    d = dict.fromkeys(['Diagnosis', 'History', 'Findings'], None)

    # ToDo: expand Diagnosis harvesting to other sources.
    if 'medpix' in x['journal_title'].lower():
        d = mexpix_info_extract(x['abstract'])

    # Define string to use when trying to harvest sex and age information.
    guess_string = x['abstract'].lower() if d['History'] is None else d['History'].lower()

    # Guess Sex
    d['sex'] = patient_sex_guess(guess_string)

    # Guess Age
    d['age'] = patient_age_guess(guess_string)

    return d


# ---------------------------------------------------------------------------------------------
# Construct Database
# ---------------------------------------------------------------------------------------------


# Pull Data
df = openi_kinesin(search_query, n_bounds_limit=5)

# Post Pull Processing
pp = pd.DataFrame(df.apply(flag_extract, axis=1).tolist()).fillna(np.NaN)

# Join df with pp
df = df.join(pp, how='left')

# Start tqdm
tqdm.pandas(desc="status")


# ---------------------------------------------------------------------------------------------
# Image Harvesting
# ---------------------------------------------------------------------------------------------


def img_harvest(image_address, db_mapping, save_location='images/', lag=5.5):
    """Harvest Pics from a URL and save to disk"""
    # Save location
    save_address = "".join(map(str, [save_location, db_mapping, "__", image_address.split("/")[-1]]))

    # Check if the file already exists
    # If not, download and save it.
    page = None
    if not Path(save_address).is_file():
        try:
            # Get the image
            page = requests.get(image_address)
        except:
            return False

        # Sleep
        sleep(abs(lag + np.random.normal()))

        # Save
        with open(save_address, 'wb') as img:
            img.write(page.content)

    return True


def bulk_img_harvest(data_frame, col1, db_map, log_save_loc="images/logs/", save_interval=25):
    """

    Bulk download of a set of images from the database

    :param data_frame:
    :param col1:
    :param db_map: column that can be used to map images back to a row in the database.
    :param log_save_loc:
    :param save_interval:
    :return:
    """
    # Download the images
    save_name = lambda: datetime.now().strftime("%d-%m-%Y_%H:%M:%S_%s")
    result_log = list()
    for x in tqdm(zip(data_frame[col1], data_frame[db_map])):
        result_log.append((x[1], img_harvest(x[0], db_mapping=x[1])))
        if x[1] % save_interval == 0:
            pd.DataFrame(result_log, columns=[db_map, 'sucessful']).to_csv(log_save_loc + save_name() + "img_log.csv")


# ---------------------------------------------------------------------------------------------
# Pull Pictures and Represent them as numpy arrays.
# ---------------------------------------------------------------------------------------------


# Run the Harvesting tool
bulk_img_harvest(df, col1='imglarge', db_map='req_no')





















































