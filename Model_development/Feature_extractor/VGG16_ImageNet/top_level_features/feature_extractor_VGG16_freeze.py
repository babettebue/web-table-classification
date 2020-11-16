# feature extraction VGG16

import pandas as pd
import numpy as np
import os as os
import re as re
from numpy import expand_dims
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16


# create vgg16 model w imgnet weights
model5 = VGG16(weights='imagenet',
               include_top=False,
               pooling='avg')
model5.summary()

# load id list per folder


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        ret = []
        for f in filenames:
            ret.append(os.path.abspath(os.path.join(dirpath, f)))
        return ret

# path_l=r'E:\Babette\MasterThesis\GoldStandard_tvt\train\0_layout'
# path_g=r'E:\Babette\MasterThesis\GoldStandard_tvt\train\1_Genuine'
# path_lv=r'E:\Babette\MasterThesis\GoldStandard_tvt\validation\0_layout'
# path_gv=r'E:\Babette\MasterThesis\GoldStandard_tvt\validation\1_Genuine'
# path_lt=r'E:\Babette\MasterThesis\GoldStandard_tvt\test\0_layout'
# path_gt=r'E:\Babette\MasterThesis\GoldStandard_tvt\test\1_Genuine'
path_l = r'GoldStandard_tvt/train/0_layout'
path_g = r'GoldStandard_tvt/train/1_Genuine'
path_lv = r'GoldStandard_tvt/validation/0_layout'
path_gv = r'GoldStandard_tvt/validation/1_Genuine'
path_lt = r'GoldStandard_tvt/test/0_layout'
path_gt = r'GoldStandard_tvt/test/1_Genuine'
train1 = absoluteFilePaths(path_l)
train2 = absoluteFilePaths(path_g)
train3 = absoluteFilePaths(path_lv)
train4 = absoluteFilePaths(path_gv)
train = train1 + train2 + train3 + train4
test1 = absoluteFilePaths(path_lt)
test2 = absoluteFilePaths(path_gt)
test = test1 + test2
dataset = train + test

#################################################################################################################

# extract features

data = []
ids = []
for name in dataset:
    # get id
    id = re.findall('id_\d*', name)
    id = int(re.sub(r'id_', '', id[0]))
    ids.append(id)

    img1 = load_img(name, target_size=(224, 224))
    img = img_to_array(img1)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = model5.predict(img)
    data.append(np.array(features[0]))


print(len(data))
# create dataframe
df_features = pd.DataFrame(data)
df_features['id'] = ids
st = []
for x in range((len(df_features))):
    if x < len(train):
        st.append('train')
    else:
        st.append('test')
df_features['set'] = st


# combine with manually created features
df_man = pd.read_pickle(r'predictions2-with-features.pkl')
#df_man= pd.read_pickle(r'E:\Babette\MasterThesis\Classifier_Dresden\predictions2-with-features.pkl')

df_full = pd.merge(df_man, df_features, on='id',  how='left')

df_full.to_pickle(r'manual_vgg16_features_dataset.pkl', protocol=4)
