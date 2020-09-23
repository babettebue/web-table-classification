
# ### Extend GS using 2020 CC

# To do in this script:
# * sample warc file paths march/april 2020
# * get html code from sampled paths
# * sample websites
# * get tables from websites
# * sample one table per website
# * filter language
# * filter out tables with less than two rows or columns

import os
import sqlite3
import pandas as pd
import imgkit
import re
import json
import boto3
import botocore
from bs4 import BeautifulSoup, Doctype
from warcio import ArchiveIterator
from langdetect import detect
import urllib.request
import numpy as np
import pickle
import random
import itertools
import logging.config
import glob
import yaml
from functools import wraps
import multiprocessing
import time
from datetime import datetime
from deco import *

ITERIM_RESULTS_FILE_PREFIX = 'gs_new_log_part'

def run():
    # passing results from one processing step to the next is done with pickle files
    # to persist interim results and support concurrency with a multithreading worker pool

    # extraction
    warc_paths = sample_random_warc_paths("warc.paths", 125)
    extract_html_from_cc(warc_paths)
    
    # filtering
    run_concurrent(extract_tables_from_html, '_extracted', '_filtered1')
    run_concurrent(filter_tables_by_language, '_filtered1', '_filtered2')
    run_concurrent(filter_out_layout_tables, '_filtered2', '_filtered3')

    # results
    _merge_files('_filtered3', '_final')


######################## DECORATOR HELPER FUNCTIONS BELOW ########################

def timing(f):
    # utils function to measure time of individual processing steps
    @wraps(f)
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.info('#### TIMING RESULT: {:s} function took {:.3f}s'.format(f.__name__, (time2-time1)))
        return ret
    return wrap


# instead @concourrency and @synchronized decorators from deco are used
# def multi_core(func):
#     def wrap(*args, **kwargs):
#         try:
#             pool = multiprocessing.Pool(4)
#             pool.map(func, *args)

#             logger.info(f'Started multiprocessing worker pool for {func.__name__} with args {args} and kwargs {kwargs}.')
#         finally: # Ensure closing processes
#             pool.close()
#             pool.join()
#         return
#     return wrap


######################## EXTRACTION FUNCTIONS BELOW ########################

def sample_random_warc_paths(file, n):
    #read warc file paths from CC March/April 2020
    with open(file, "r") as f:
        paths = f.read()

    p = paths.split('\n')

    # take random sample of paths
    return random.sample(p, n)



def get_s3_client(creds_dir=''):
    # creds_dir = r'C:\Users\babet\Documents\Studi\Master Thesis\Table_detection'

    # import credentials for aws server
    creds_file = os.path.join(creds_dir, 'credentials.json')
    with open(creds_file) as creds:
        credentials = json.load(creds)

    # use boto3
    session = boto3.Session(
        aws_access_key_id=credentials['aws_access_key'],
        aws_secret_access_key=credentials['aws_secret_key'],
    )
    return session.resource('s3')


@timing
def extract_html_from_cc(warc_path_sample):
    try:
        # spawns as many processes as cpu cores available: cpu_count()
        pool = multiprocessing.Pool()
        pool.map(_extract_html_from_cc_for_warc_path, warc_path_sample)
        logger.info('Started multiprocessing worker pool for _extract_html_from_cc_for_warc_path')

    finally:
        # Ensure closing processes
        pool.close()
        pool.join()


@timing
def _extract_html_from_cc_for_warc_path(warc_path):
    logger = _get_logger()
    logger.info(f'Extracting websites from common crawl for warc path {warc_path}.')

    html = []
    url = []
    s3link = []
    s3 = get_s3_client()
    obj = s3.Object(bucket_name='commoncrawl', key=warc_path)
    response = obj.get()
    data = response['Body']

    logger.info(f'Start iterating over extracted common crawl websites...')
    for record in ArchiveIterator(data):
        if record.rec_type == 'response':
                if record.http_headers.get_header('Content-Type') == 'text/html':
                    html.append(record.content_stream().read())
                    url.append(record.rec_headers.get_header('WARC-Target-URI'))
                    s3link.append(warc_path)

    logger.info(f'Successfully extracted {len(html)} websites')

    # make dataset
    d = {'url': url, 'html_full': html, 's3Link': s3link}
    df = pd.DataFrame(data=d)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    _persist_html_df(df, file_suffix=f'_{timestamp}_extracted')
    logger.info(f'Successfully persisted html pickle file with timestamp {timestamp}')


######################## CONCURRENCY UTIL FUNCTIONS BELOW ########################

@synchronized
def run_concurrent(func, input_suffix, result_suffix):
    files = glob.glob(f'{ITERIM_RESULTS_FILE_PREFIX}*{input_suffix}.pkl')

    logger.info(f'Iterating over files with suffix {input_suffix} to concurrently run {func.__name__}.')
    for file in files:
        run_with_interim_results_persisting(func.__name__, file, result_suffix)

@concurrent
def run_with_interim_results_persisting(func_name, file, result_suffix):
    logger.info(f'Calling {func_name} for file {file}.')
    df = pd.read_pickle(file)

    result = globals()[func_name](df)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    _persist_html_df(result, file_suffix=f'_{timestamp}{result_suffix}')
    logger.info(f'Successfully persisted pickled results for {func_name} with suffix {result_suffix} and timestamp {timestamp}.')


######################## FILTERING FUNCTIONS BELOW ########################

def extract_tables_from_html(df, sample=None):
    logger.info('Filtering websites with tables.')

    if isinstance(sample, int):
        df_s = df.sample(n=sample, random_state=1)
        logger.info(f'Sampled {sample} websites per warc file.')
    else:
        df_s = df.copy()

    df_s.reset_index(inplace=True)
    df_s = df_s.drop(columns= ['index'])

    # count nbr of tables per html
    tables = []
    tab_num = []
    titles = []
    texts = []

    for html in df_s.html_full:
        try:
            bs = BeautifulSoup(html, 'html.parser')
            website_tables = bs.find_all("table")
            num = len(website_tables)

            if num == 0:
                raise Exception('No tables found on website')

            i = random.randint(1, num)
            table_text = re.sub("\n", "", str(website_tables[i-1].getText()))
            selected_table = str(website_tables[i-1])
            title_2 = re.sub(r'<title>', r'', str(bs.title))
            title_3 = re.sub(r'</title>', r'', title_2)
            title_4 = re.sub(r'\n', r'', title_3)
            title_5 = re.sub(r'\r', r'', title_4)
        except Exception as e:
            logger.warning(f'Failed to parse website html due to error: {e}. Skipping website.')
            i = 0
            table_text = ''
            selected_table = ''
            title_5 = ''

        tab_num.append(i)
        tables.append(selected_table)
        texts.append(table_text)
        titles.append(title_5)

    df_s["tab_num"] = tab_num
    df_s["table"] = tables
    df_s["tableText"] = texts
    df_s["pageTitle"] = titles

    #only keep htmls with tables
    df_s = df_s.drop(df_s[df_s.tab_num == 0].index)
    df_s = df_s.reset_index()
    df_s = df_s.rename(columns={"html_full": "fullHtmlCode", "tab_num": "tableNum", "table": "htmlCode"})
    df_s = df_s[['url','fullHtmlCode','tableNum','htmlCode', 's3Link', 'pageTitle', 'tableText']]

    logger.info('Successfully filtered websites with tables.')
    return df_s


def filter_tables_by_language(df, language='en'):
    logger.info(f'Filtering tables by language {language}.')

    # detect language
    lang = []
    for i in df.tableText:

        try:
            lang.append(detect(i))
        except:

            lang.append("NAN")
    lang2 = []
    for i in df.pageTitle:

        try:
            lang2.append(detect(i))
        except:
            lang2.append("NAN")


    df["languageTable"] = lang
    df["languageTitle"] = lang2
    df["language"] = np.where(df["languageTable"] == "NAN", df["languageTitle"], df["languageTable"])

    logger.info(f'Languages in html tables:\n {df["language"].describe()}')

    # Keep english tables only
    filtered_df = df.loc[df['language'] == language]

    logger.info(f'Successfully filtered websites with {language} tables.')
    return filtered_df


def filter_out_layout_tables(df_s):
    # ### apply simple filter rules

    # * remove tables with less than 2 rows
    # * remove tables with less than 2 columns

    logger.info('Filtering out layout tables.')

    filt = []
    for i, html in df_s.fullHtmlCode.iteritems():
        bs = BeautifulSoup(html, 'html.parser')

        #find respective table by table number on website
        num = df_s.tableNum[i]-1
        table = bs.find_all(re.compile("table"))[num]

        #find and count rows
        rows = table.find_all("tr")
        if len(rows) < 2:
            filt.append(False)
            continue

        #count columns in second row
        n_columns = len(rows[1].find_all("td"))
        if n_columns < 2:
            filt.append(False)
            continue

        filt.append(True)

    #apply filter
    df_final = df_s[filt]
        # filtered_list_df.append(df_final)
        # filt_all.extend(filt)

    share = filt.count(False) * 100 / (filt.count(True) + filt.count(False))

    logger.info('Successfully filtered layout tables. Filtering results:')
    logger.info(f"Candidate tables: {filt.count(True)}")
    logger.info(f"Layout tables: {filt.count(False)}")
    logger.info(f"Share of filtered tables: {round(share,2)}")

    return df_final


######################## HELPER FUNCTIONS BELOW ########################

def _get_logger(log_config_path=''):
    log_config_file = os.path.join(log_config_path, 'logger_config.yml')
    with open(log_config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        logging.config.dictConfig(config)
        logger = logging.getLogger(__name__)
        logger.debug('Logger initialized.')
    return logger


def _merge_files(input_suffix, result_suffix):
    files = glob.glob(f'{ITERIM_RESULTS_FILE_PREFIX}*{input_suffix}.pkl')
    logger.info(f'Merging the following files into one {files}')

    # loading data
    collection = {}
    for file in files:
        collection[file] = pd.read_pickle(file)

    # combining into one dataset
    df_final = pd.concat(collection.values())
    df_final = df_final.reset_index()
    df_final['id'] = df_final.index

    # persisting
    df_final = df_final[['id','url','fullHtmlCode','tableNum','htmlCode', 's3Link', 'pageTitle', 'tableText', 'language', 'languageTitle', 'languageTable']]
    _persist_html_df(df_final, file_suffix=result_suffix)
    logger.info(f'Successfully persisted single, merged file with suffix {result_suffix}')


def _persist_html_df(df, file_suffix='', path_dir='', splits=1):
    df_arr = np.array_split(df, splits)

    for n in range(splits):
        file_name = os.path.join(path_dir, f'{ITERIM_RESULTS_FILE_PREFIX}{n}{file_suffix}.pkl')
        df_arr[n].to_pickle(file_name)
    logger.info(f'Successfully persited websites in {splits} pkl file(s).')


def _load_html_df(file_suffix='', path_dir='', splits=None):
    if splits is None:
        pattern = os.path.join(path_dir, f'{ITERIM_RESULTS_FILE_PREFIX}*{file_suffix}.pkl')
        logger.debug(f"Param 'splits' not provided. Globing all files with pattern '{pattern}'.")
        files = glob.glob(pattern)
    else:
        files = [os.path.join(path_dir, f'{ITERIM_RESULTS_FILE_PREFIX}{n}{file_suffix}.pkl') for n in range(splits)]

    df_arr = []
    for file in files:
        df_arr.append(pd.read_pickle(file))

    df = pd.concat(df_arr)
    logger.info(f'Successfully loaded {len(df)} websites from pkl files.')
    return df


def _pickle_html_iterator(file_suffix='', path_dir=''):
    file_name = os.path.join(path_dir, f'{ITERIM_RESULTS_FILE_PREFIX}*{file_suffix}.pkl')
    files = glob.glob(file_name)
    logger.info(f'Found following pickle files: {files}')

    for file in files:
        df = pd.read_pickle(file)
        logger.info(f'Successfully loaded {len(df)} websites for pkl file {file}.')
        yield df


def _split_df_by_s3link(df):
    path_unique = df["s3Link"].unique()

    list_df = []
    for path in path_unique:
        list_df.append(df[path==df.s3Link])

    logger.info(f'Successfully split dataframe by s3Link in {len(path_unique)} subsets.')
    return list_df


######################## MAIN ########################

if __name__ == '__main__':
    logger = _get_logger()
    run()