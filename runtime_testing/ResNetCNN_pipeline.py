#Pipeline to classify webtables with ResNet CNN

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from utils.image_rendering import timing, html_to_images, timing_html_to_images

# define paths:
resources = os.path.join('runtime_testing', 'resources')
dataset_path = os.path.join(resources, 'performance_testing_5_tables.pkl')
weights_path = os.path.join(resources, 'resnet_model1_finetuned.h5')
#load performace testing data consisting of html code of 100 web pages and table number indicating which table on the page should be classified and the website url

def run():
    #load datarframe containing ["fullTable]        table code
    #                           ["fullHtmlCode"]    website code
    #                           ["url"]             website url
    df = pd.read_pickle(dataset_path)

    #returns integer predictions of image classification with finetuned ResNet50 model
    ids, predictions = make_prediction(df, weights_path)

@timing
def make_prediction(df, weights_path):
    ids, images = timing_html_to_images(df, 224)

    #create tensorflow dataframe
    tf.compat.v1.disable_eager_execution()
    tf_df = tf.data.Dataset.from_tensor_slices(np.asarray(images))

    #load resnet model with finetuned weights
    resnet = load_resnet_ft(weights_path)

    #make prediction
    predictions = np.around(resnet.predict(tf_df)).astype(int)

    return ids, predictions


def load_resnet_ft(weights_path):
    resnet_model = ResNet50(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling=None)
    for layer in resnet_model.layers:
        layer.trainable = True

    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(224, 224, 3)))
    model.add(resnet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))

    model.load_weights(str(weights_path))

    return model


if __name__ == '__main__':
    run()
