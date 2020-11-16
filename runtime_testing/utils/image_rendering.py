from tensorflow.keras.preprocessing.image import img_to_array
from numpy import expand_dims
from bs4 import BeautifulSoup, Doctype
from pathlib import Path
import pandas as pd
import csv
import re
from deco import *
import time
import imgkit
from sys import platform
from functools import wraps
import io
from PIL import Image
import numpy as np
import os
import csv

path_wkhtmltoimage = os.path.join('runtime_testing', 'resources', 'wkhtmltox-0.12.5-1.mxe-cross-win64', 'wkhtmltox', 'bin', 'wkhtmltoimage.exe')

def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start_time = time.time()
        ret = f(*args, **kwargs)
        time_diff = "{:.3f}".format(time.time() - start_time)
        print(f'\n############ BENCHMARKING RESULT ############\n||||| {f.__module__} - {f.__name__} function took {time_diff}s\n')
        with open("timing.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow([f.__module__, f.__name__, time_diff])
        return ret
    return wrap


@timing
def timing_html_to_images(df, image_size=224):
    return html_to_images(df, image_size)


@synchronized
def html_to_images(df, image_size=224):

    #prepare table html code for rendering
    df1 = extract_relevant_html_parts(df)
    df2 = create_renderable_code(df1)

    #render image
    rendering_results = {}
    for index, row in df2.iterrows():
        rendering_results[row['id']] = render_image(row['renderCodeLink'], image_size)

    #filter out unsuccessful image renderings
    rendering_successes = {k: v for k, v in rendering_results.items() if v is not None}
    images = list(rendering_successes.values())
    ids = list(rendering_successes.keys())
    
    return ids, images


@concurrent
def render_image(html, image_size):
    img_pix = None
    config, options = get_imgkit_configuration(platform, path_wkhtmltoimage)
    try:
        img = imgkit.from_string(html, False, config=config, options=options)
        t_file = io.BytesIO(img)
        img_pil = Image.open(t_file)
        img_pil_size = img_pil.resize((image_size, image_size))
        img_ar = img_to_array(img_pil_size)
        img_pix = expand_dims(img_ar, axis=0)
    except Exception as e:
        print(f'Image rendering failed: {e}')
    return img_pix


def get_imgkit_configuration(platform, path_wkhtmltoimage=None):
    config = None
    options = {'quiet': '', 'load-error-handling': 'ignore',
               'load-media-error-handling': 'ignore'}

    if platform == "linux" or platform == "linux2":
        options['xvfb'] = ''
    elif platform == "darwin":
        pass
    elif platform == "win32":
        if path_wkhtmltoimage is None:
            raise Exception('Please specify the path to the wkhtmltoimage exe.')
        config = imgkit.config(wkhtmltoimage=path_wkhtmltoimage)

    return config, options


def extract_relevant_html_parts(sdf):
    # Extract relevant html parts
    # * Extract table code
    # * Extract links to css files
    # * Complete relative links
    # * Extract style tags

    links = []
    style = []
    table = []

    #for html in sdf.fullHtmlCode:
    for index, row in sdf.iterrows():
        try:
            bs = BeautifulSoup(row['fullHtmlCode'], 'html.parser')

            tables = bs.find_all("table")
            tableString = str()
            tableString = str(tables[row['tableNum']-1])

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
            tableString = ''

        links.append(longString)
        style.append(styleString)
        table.append(tableString)

    sdf['links'] = links
    sdf['styleTag'] = style
    sdf['fullTable'] = table
    return sdf


def create_renderable_code(sdf):

    renderCode = []

    for _, row in sdf.iterrows():
        #row= df1.iloc[1]
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


def pickle_to_formatted_csv(pickle_file):
    pickle_path = Path(pickle_file)
    csv_path = str(pickle_path.with_suffix('.csv'))
    df = pd.read_pickle(pickle_file)
    df_s = df[['id', 'fullTable']]
    df_s['fullTable'] = df_s['fullTable'].replace('\n|\r|\t','', regex=True) 
    df_s.to_csv(csv_path, header=True, doublequote=False, quoting=csv.QUOTE_NONE, escapechar="\\", index=False)
    return csv_path