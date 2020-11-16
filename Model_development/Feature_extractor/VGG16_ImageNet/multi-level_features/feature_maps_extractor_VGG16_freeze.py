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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D


# create vgg16 model w imgnet weights
model5 = VGG16(weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(225,225,3))
model5.summary()

model1 = Sequential()
model1.add(keras.Input(shape=(224,224,3)))
for layers in model5.layers[1:-16]:
    model1.add(layers)
model1.add(GlobalAveragePooling2D())
model1.summary()


model2 = Sequential()
model2.add(keras.Input(shape=(224,224,3)))
for layers in model5.layers[1:7]:
    model2.add(layers)
model2.add(GlobalAveragePooling2D())
model2.summary()


model3 = Sequential()
model3.add(keras.Input(shape=(224,224,3)))
for layers in model5.layers[1:11]:
    model3.add(layers)
model3.add(GlobalAveragePooling2D())
model3.summary()

model4 = Sequential()
model4.add(keras.Input(shape=(224,224,3)))
for layers in model5.layers[1:15]:
    model4.add(layers)
model4.add(GlobalAveragePooling2D())
model4.summary()

#load id list per folder
def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
        ret=[]
        for f in filenames:
           ret.append(os.path.abspath(os.path.join(dirpath, f)))
        return ret

# path_l=r'E:\Babette\MasterThesis\GoldStandard_tvt\train\0_layout'
# path_g=r'E:\Babette\MasterThesis\GoldStandard_tvt\train\1_Genuine'
# path_lv=r'E:\Babette\MasterThesis\GoldStandard_tvt\validation\0_layout'
# path_gv=r'E:\Babette\MasterThesis\GoldStandard_tvt\validation\1_Genuine'
# path_lt=r'E:\Babette\MasterThesis\GoldStandard_tvt\test\0_layout'
# path_gt=r'E:\Babette\MasterThesis\GoldStandard_tvt\test\1_Genuine'
path_l=r'GoldStandard_tvt/train/0_layout'
path_g=r'GoldStandard_tvt/train/1_Genuine'
path_lv=r'GoldStandard_tvt/validation/0_layout'
path_gv=r'GoldStandard_tvt/validation/1_Genuine'
path_lt=r'GoldStandard_tvt/test/0_layout'
path_gt=r'GoldStandard_tvt/test/1_Genuine'
train1= absoluteFilePaths(path_l)
train2= absoluteFilePaths(path_g)
train3= absoluteFilePaths(path_lv)
train4= absoluteFilePaths(path_gv)
train = train1 + train2 + train3 + train4
test1= absoluteFilePaths(path_lt)
test2= absoluteFilePaths(path_gt)
test= test1 + test2
dataset = train + test

#################################################################################################################


#extract features, from all conv blocks!

data1=[]
data2=[]
data3=[]
data4=[]
data5=[]
ids=[]
for name in dataset:
     #get id
    id= re.findall('id_\d*', name)
    id = int(re.sub(r'id_', '', id[0]))
    ids.append(id)

    img1= load_img( name, target_size=(224,224))
    img= img_to_array(img1)
    img= expand_dims(img, axis=0)
    img= preprocess_input(img)

    
    features1 = model1.predict(img)
    data1.append(np.array(features1[0]))
    features2 = model2.predict(img)
    data2.append(np.array(features2[0]))
    features3 = model3.predict(img)
    data3.append(np.array(features3[0]))
    features4 = model4.predict(img)
    data4.append(np.array(features4[0]))
    features5 = model5.predict(img)
    data5.append(np.array(features5[0]))



print(len(data1))
#create dataframe
df_features1 = pd.DataFrame(data1)
df_features1['id']= ids
df_features2 = pd.DataFrame(data2)
df_features2['id']= ids
df_features3 = pd.DataFrame(data3)
df_features3['id']= ids
df_features4 = pd.DataFrame(data4)
df_features4['id']= ids
df_features5 = pd.DataFrame(data5)
df_features5['id']= ids
st=[]
for x in range((len(df_features5))):
    if x < len(train):
        st.append('train')
    else: st.append('test')
df_features5['set']= st

#combine with manually created features
df_man= pd.read_pickle(r'predictions2-with-features.pkl')
#df_man= pd.read_pickle(r'E:\Babette\MasterThesis\Classifier_Dresden\predictions2-with-features.pkl')
print(df_man.columns)
print(df_features5.columns)

df_full= pd.merge(df_man, df_features1, on='id',  how='left')
df_full= pd.merge(df_full, df_features2, on='id',  how='left', suffixes=('_conv1', '_conv2'))
df_full= pd.merge(df_full, df_features3, on='id',  how='left', suffixes=('_conv2', '_conv3'))
df_full= pd.merge(df_full, df_features4, on='id',  how='left', suffixes=('_conv3', '_conv4'))
df_full= pd.merge(df_full, df_features5, on='id',  how='left', suffixes=('_conv4', '_conv5'))

df_full.to_pickle(r'manual_vgg16_feature_maps_dataset.pkl', protocol=4)
