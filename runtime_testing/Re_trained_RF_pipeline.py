#Pipeline to classify webtables re-trained Random Forest  DWTC manual features 

import pandas as pd
import numpy as np
import os
import pickle
from joblib import load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
from utils.image_rendering import timing
from utils.dwtc_wrapper import calculate_manual_features

# define paths:
resources = os.path.join('runtime_testing', 'resources')
dataset_path = os.path.join(resources, 'performance_testing_100_tables.pkl')
model_path = os.path.join(resources, 'heuristic_rf.joblib')
random_forest_model_path = os.path.join(resources, 'RandomForest_P1.mdl')

#load performace testing data consisting of html code of 100 web pages and table number indicating which table on the page should be classified and the website url

def run():
    #returns integer predictions of image classification with manual and visual features random forest
    ids, predictions = make_prediction(dataset_path, model_path, random_forest_model_path)


@timing
def make_prediction(dataset_path, model_path, random_forest_model_path):

    #calculate manual features based on HTML code with use of DWTC-extractor
    man_feat = calculate_manual_features(dataset_path, random_forest_model_path)
    ids = man_feat['id'].tolist()
    man_feat.drop(['id'], axis=1, inplace=True)


    #load Random Forest model
    rf = load(model_path)
    predictions= rf.predict(man_feat)

    return ids, predictions


if __name__ == '__main__':
    run()
