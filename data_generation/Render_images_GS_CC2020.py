
# coding: utf-8

# ## Render web table images for new GS 
#
# * Extract relevant html parts: css-links style tags and original table code
# * Get Wayback Machine Internet Archive Link to access archived CSS files
# * Create renderable html code
# * Render and save images


import pathlib
import os
import pandas as pd
import imgkit
import re
import json
import boto3
import botocore
from bs4 import BeautifulSoup, Doctype
from warcio import ArchiveIterator
from langdetect import detect
import requests
import numpy as np
import pickle
import random
import itertools
import logging.config
import glob
import yaml
from functools import wraps
import multiprocessing
import uuid
import time
from datetime import datetime
from sys import platform
from deco import *

def run():
    # load ne gold standard dataset
    df = pd.read_pickle(r'final_result_125_warc_files.pkl')
	
    # prepare html code for rendering
    ######################################################################################################################################
    prepare1 = run_concurrent(extract_relevant_html_parts, df)  # exract link and style tags
    #prepare1 = extract_relevant_html_parts(df)
    # renderCoder & renderCodeLink (completed links)
    prepare2 = run_concurrent(create_renderable_code, prepare1)
    #prepare2 = create_renderable_code(prepare1)
    # # renderCodeWayback (including wayback paths)
    prepare2.to_pickle(r'gs_prepare2_renderable_125_warc_files.pkl')
    #prepare2 = pd.read_pickle('gs_prepare2_renderable_250_warc_files.pkl')
    prepare3 = run_concurrent(add_wayback_machine_paths, prepare2, timestamp='20200401')
    #prepare3.to_pickle('gs_renderable_250_warc_files.pkl')
    prepare3 = pd.read_pickle(r'gs_renderable_125_warc_files.pkl')

	#####################################################################################################################################
    # render Images
    img_dir_prefix = 'rendered-images'
    img_dir = img_dir_prefix + '-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    error = run_concurrent(render_images, prepare3, img_dir=img_dir)  # use wayback paths y/n
    error.to_pickle('gs_250_wf_render_error.pkl')

########################################## prepare html code for rendering ###################################################################


def extract_relevant_html_parts(df_s):
    # Extract relevant html parts
    # * Extract links to css files
    # * Complete relative links
    # * Extract style tags

    sdf = df_s
    sdf = sdf.rename(columns={"htmlCode": "fullTable"})
    links = []
    style = []

    for html in sdf.fullHtmlCode:
        try:
            bs = BeautifulSoup(html, 'html.parser')
            head = bs.find_all("head")
            # css links
            link = head[0].find_all("link")
            longString = str()
            for x in link:
                longString = longString + " \n " + str(x)

            # style tags
            sty = head[0].find_all('style')
            styleString = str()
            for x in sty:
                styleString = styleString + " \n " + str(x)

        except Exception as e:
            #logger.warning(f'Failed to parse website html due to error: {e}. Skipping website.')

            longString = ''
            styleString = ''

        links.append(longString)
        style.append(styleString)

    sdf['links'] = links
    sdf['styleTag'] = style
    return sdf


def create_renderable_code(sdf):

    renderCode = []

    for _, row in sdf.iterrows():
        links = row['links']
        styleTag = row['styleTag']
        fullTable = row['fullTable']
        # Create Table html code including links & style info
        htmlAdd = r'<html> <head> <meta charset="UTF-8" /> ' + links + \
            styleTag + r' </head> <body> ' + fullTable + r' </body> </html>'
        renderCode.append(htmlAdd)
    sdf['renderCode'] = renderCode

    # create complete links if necessary
    links_comp = []

    for _, row in sdf.iterrows():
        url = re.sub('/$', r'', row['url'])
        # print(url)

        # case 1 current directory
        links1 = re.sub('href="(?!(/|\.\.|http))', r'href="' +
                        url + r'/', row['renderCode'])

        # case 2: current root
        root_url = re.match('https?://(?:.*\.)*(.+\..+?)/',
                            row['url']).group(0)
        #short_url= re.sub(r"/(?:.(?!/))+$", r"/", url)
        # print(short_url)
        links2 = re.sub('href="/(?!/)', r'href="' + root_url, links1)

        # case 3: two above
        shorter_url = re.sub("/[^/]*/(?:.(?!/))+$", r"/", url)
        links3 = re.sub('href="../', r'href="' + shorter_url, links2)
        # print(shorter_url)

        # case4: autonomous link //
        links4 = re.sub('href="//', r'href="http://', links3)

        # same for style links

        # case 1 current directory
        links5 = re.sub(
            'url\((\\\')(?!(/|\.\.|\"http|http|https|\"https))', r'url(\'' + url + r'/', links4)
        links6 = re.sub(
            'url\((?!(/|\.\.|\"http|http|https|\"https|\\\'))', r'url(' + url + r'/', links5)

        # case 2: current root
        #short_url= re.sub(r"/(?:.(?!/))+$", r"/", url)
        # print(short_url)
        links7 = re.sub('url\(\\*\'*/(?!/)', r'url(\'' + root_url, links6)
        links8 = re.sub('url\(/(?!/)', r'url(' + root_url, links7)

        # case 3: two above
        shorter_url = re.sub("/[^/]*/(?:.(?!/))+$", r"/", url)
        links9 = re.sub('url\(\\*\'*\.\./', r'url(\'' + shorter_url, links8)
        links10 = re.sub('url\(\.\./', r'url(' + shorter_url, links9)
        # print(shorter_url)

        # case 4 autonomous link
        links11 = re.sub('url\(//', r'url(http://', links10)

        # caseX scr
        links12 = re.sub(
            'src=\"(?!(/|\.\.|\"http|http|https|\"https|\\\'))', r'src="' + url + r'/', links11)
        links13 = re.sub('src=\"/(?!/)', r'src="' + root_url, links12)
        links14 = re.sub('src=\"\.\./', r'src="' + shorter_url, links13)
        links15 = re.sub('src=\"//', r'src="http://', links14)

        # links6=links3
        links_comp.append(links15)

    sdf['renderCodeLink'] = links_comp

    return sdf


def add_wayback_machine_paths(sdf, timestamp):
    # Get wayback machine path to access css files
    archivePath = []

    for _, row in sdf.iterrows():
        req_string = "http://web.archive.org/wayback/available"
        params = {'url': row['url'], 'timestamp': timestamp}
        try:
            response = requests.get(req_string, params, timeout=5)
            if response.status_code == 200:
                dic = response.json()
                if 'closest' in dic['archived_snapshots']:
                    archive_url = dic['archived_snapshots']['closest']['url']
                    archive = re.match(r'http://web.archive.org/web/\d*/', archive_url)
                    archivePath.append(str(archive.group(0)))
                    continue

        except Exception as e:
            print(f'An error occured getting wayback machine path for {req_string}: {e}')

        archivePath.append("NAN")

    sdf['archivePath'] = archivePath

    # add paths to archived css documents to renderable Code
    renderCodeWb = []

    for _, row in sdf.iterrows():
        archivePath = row['archivePath']
        html = row['renderCodeLink']

        if archivePath != 'NAN':
            htmlA = re.sub(r'"http:', r'"' + archivePath + 'http:', html)
            htmlArchive = re.sub(
                '\(http:', '\(' + archivePath + 'http:', htmlA)
        else:
            htmlArchive = html
        renderCodeWb.append(htmlArchive)

    sdf['renderCodeWayback'] = renderCodeWb

    return sdf


###################################### concurrency utils ###############################################################
def _split_df_by_s3link(df):
    path_unique = df["s3Link"].unique()

    list_df = []
    for path in path_unique:
        list_df.append(df[path==df.s3Link])

    print(f'Successfully split dataframe by s3Link in {len(path_unique)} subsets.')
    return list_df


@synchronized
def run_concurrent(func, df, *args, **kwargs):
    df_list = _split_df_by_s3link(df)
    print(f'Iterating over dataframes splitted by S3 link to concurrently run {func.__name__}.')
    results = {}
    for df_split in df_list:
        s3link = df_split['s3Link'].iloc[0]
        print(f'Running func {func.__name__} for s3link {s3link}')
        results[s3link] = concurrency_wrapper(func.__name__, df_split, *args, **kwargs)

    print('Trying to combine all results into single dataframe.')
    return pd.concat(results.values())


@concurrent
def concurrency_wrapper(func_name, *args, **kwargs):
    return globals()[func_name](*args, **kwargs)


###################################### render images ###############################################################
def render_images(df, img_dir=None, img_dir_prefix='rendered-images', wkhtml_p=None, wb=False):

    config = None
    options = {'quiet': ''}
    if platform == "linux" or platform == "linux2":
        options['xvfb'] = ''
    elif platform == "darwin":
        pass
    elif platform == "win32":
        if wkhtml_p is None:
            raise Exception('Please specify the path to the wkhtmltoimage exe.')
        config = imgkit.config(wkhtmltoimage=wkhtml_p)

    if img_dir is None:
        img_dir = img_dir_prefix + '-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

    render_error = []
    render_error_msg = []

    for _, row in df.iterrows():
        path = os.path.join(img_dir, f"id_{row['id']}.jpg")
        if wb == True:
            # include wayback machine links:
            html = row['renderCodeWayback']
        else:
            html = row['renderCodeLink']
            #html = row['fullHtmlCode']

        try:
            imgkit.from_string(html, path, options=options, config=config)
            render_error.append(0)
            render_error_msg.append('')
            print(f"Successfully rendered image with id {row['id']}.")
        except Exception as e:
            print(f"An error occured when rendering image with id {row['id']}: {e}")
            # traceback.print_exc() # import traceback
            render_error.append(1)
            render_error_msg.append(str(e))

    #df["render_error_archive"]= render_error

    error_df = pd.DataFrame({'id': df.id, 'render_error': render_error, 'render_error_msg': render_error_msg})
    s3link = df['s3Link'].iloc[0]
    persist_errors(error_df, img_dir, s3link)
    return error_df

def persist_errors(error_df, img_dir, s3link):
    try:
        warc_name = get_warc_file_name(s3link)
        path = os.path.join(img_dir, 'rendering-errors')
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        file = os.path.join(path, f'errors_{warc_name}.csv')
        error_df.to_csv(file, sep='\t')
    except Exception as e:
        print(f'An error occured when trying to persist intermediate rendering errors as a csv file: {e}')

def get_warc_file_name(s3link):
    try:
        pattern = 'warc\/(.*?)\.warc\.gz'
        warc_name = re.search(pattern, s3link).group(1)
    except Exception as e:
        print(f'An error occured when trying to extract the warc file name from the s3 link {s3link}: {e}')
        print(f'Generating random uuid for s3 link {s3link} instead...')
        warc_name = f'random-replacement-uuid-{uuid.uuid4()}'
    return warc_name


######################## MAIN ########################
if __name__ == '__main__':
 #   logger = _get_logger()
    run()
