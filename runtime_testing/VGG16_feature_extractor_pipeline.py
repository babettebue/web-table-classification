#Pipeline to classify webtables Random Forest using VGG16 visual features and DWTC manual features 

import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from utils.image_rendering import pickle_to_formatted_csv, html_to_images, timing, timing_html_to_images
from utils.dwtc_wrapper import calculate_manual_features

# define paths:
resources = os.path.join('runtime_testing', 'resources')
dataset_path = os.path.join(resources, 'performance_testing_100_tables.pkl')
model_path = os.path.join(resources, 'heuristic_vgg16_feature_maps_rf.joblib')
random_forest_model_path = os.path.join(resources, 'RandomForest_P1.mdl')

#load performace testing data consisting of html code of 100 web pages and table number indicating which table on the page should be classified and the website url

def run():
    #load datarframe containing ["fullTable]        table code
    #                           ["fullHtmlCode"]    website code
    #                           ["url"]             website url
    df = pd.read_pickle(dataset_path)

    #returns integer predictions of image classification with manual and visual features random forest
    ids, predictions = make_prediction(df, dataset_path, model_path, random_forest_model_path)

@timing
def make_prediction(df, dataset_path, model_path, random_forest_model_path):

    #calculate manual features based on HTML code with use of DWTC-extractor
    man_feat = calculate_manual_features(dataset_path, random_forest_model_path)
    
    #extract visual features from VGG16
    ids, images = timing_html_to_images(df, 225)
    vis_feat = get_features(ids, images)

    #load and join manual features
    ids, features = join_manual_features(vis_feat, man_feat)

    #load Random Forest model
    rf = load(model_path)
    predictions= rf.predict(features)

    return ids, predictions


def join_manual_features(vis_feat, man_feat):
    df_full = pd.merge(man_feat, vis_feat, on='id',
                       how='right', suffixes=['_man', '_vis'])
    ids = df_full['id'].tolist()
    df_full.drop(['id'], axis=1, inplace=True)

    #mean imputation for nested tables
    for x in df_full.iloc[:, 0:26].columns:
        df_full[x] = df_full[x].fillna(df_full[x].mean())
    print(df_full)

    return ids, df_full


def get_features(ids, images):

    #preprocess images with tensorflow
    preprocessed_images = [preprocess_input(image) for image in images]

    #create tensorflow dataframe
    tf_df = tf.data.Dataset.from_tensor_slices(np.asarray(preprocessed_images))

    #load resnet model with finetuned weights
    resnet = load_vgg16()

    #get feature maps
    pred = resnet.predict(tf_df)

    df_f = pd.DataFrame(pred[0])
    for i in range(1, 5):
        df_f = pd.concat([df_f, pd.DataFrame(pred[i])], axis=1)

    df_f['id'] = ids

    return df_f


def load_vgg16():
    vgg = VGG16(weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(225, 225, 3))

    #indeces of output layers [3, 6, 10, 14]
    conv1 = GlobalAveragePooling2D()(vgg.layers[3].output)
    conv2 = GlobalAveragePooling2D()(vgg.layers[6].output)
    conv3 = GlobalAveragePooling2D()(vgg.layers[10].output)
    conv4 = GlobalAveragePooling2D()(vgg.layers[14].output)
    outputs = [conv1, conv2, conv3, conv4, vgg.output]
    model = Model(inputs=vgg.inputs, outputs=outputs)

    return model


if __name__ == '__main__':
    run()