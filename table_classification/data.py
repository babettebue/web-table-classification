#create train and test data

import numpy as np
import tensorflow as tf
import pathlib
import cv2 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os as os
from tqdm import tqdm 
from random import shuffle 
import re

# load train and test dataset
def generate_data(data_path= '', batch= 32, height= 300, width= 300):

    #resize rgb channels
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            str(data_path),
            target_size=(height, width),
            batch_size=batch,
            class_mode='binary')
    return train_generator


#load datasets as npy arrays

def data_as_npy_arrays(data_path='', height=224, width=224, name=''):

    #data_dir = pathlib.Path(r'E:\Babette\MasterThesis\GoldStandard_tvt_sub\train')
    data_dir = pathlib.Path(str(data_path))

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(str(image_count) + " images found")


    training_data = [] 
    images_path =[]
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        for class_images in os.listdir(class_path):
            images_path.append(os.path.join(class_path, class_images))
    
    # loading the training data 
    
    for path in tqdm(images_path): 
    #for path in mages_path: 
        
        # load label from directory
        label = label_img(path) 
  
        # loading the image from the path
        img = cv2.imread(path) 

        #greyscale to minimize dimension
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # resizing the image
        img = cv2.resize(img, (width,height)) 

        #normalize rgb channels to 0,1

        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  
        #training data list with numpy array of the images 
        training_data.append([np.array(img), np.array(label)]) 

    shuffle(training_data) 

    p= str(name)+ r'_data.npy'
    np.save(p, training_data) 

    return training_data


def load_data(array=True ,dp='', n='', h=224, w=224):

    if array==True:
        train_dat = np.load(str(dp), allow_pickle= True) 
    else:
        train_dat = data_as_npy_arrays(data_path= dp, height=h, width=w, name=n)

    # X-Features & Y-Labels 
  
    x_train = np.array([i[0] for i in train_dat]).reshape(-1, h, w, 1) 
    y_train = np.array([i[1] for i in train_dat])
    y_train = np.array([i[0] for i in y_train])

    print('Shape of data tensor:', x_train.shape)
    print('Shape of label tensor:', y_train.shape)

    return x_train, y_train



def label_img(path): 
    word_label = path.split('\\')[-2] 
    if word_label == '1_Genuine': return [1, 0] 
    elif word_label == '0_layout': return [0, 1] 
    
#find avg height and width to define resizing params
def avg_img_size(resize_factor= 4, saved_file= None, data_path= ''):
    
    if saved_file== None:

        #train_dir = pathlib.Path(r'E:\Babette\MasterThesis\GoldStandard_tt\train') 
        train_dir = pathlib.Path(str(data_path)) 

        avg_height = 0
        avg_width = 0
        total_train = 0

        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            class_images = os.listdir(class_path)

            for img_name in class_images:
                h, w, c = cv2.imread(os.path.join(class_path, img_name)).shape
                avg_height += h
                avg_width += w

            total_train += len(class_images)
            
        IMG_HEIGHT= avg_height//total_train
        IMG_WIDTH= avg_width//total_train

        shape = [IMG_HEIGHT, IMG_WIDTH]
        #save values
        np.save(r'savedshape.npy', shape) 

    else:
        #load values from file
        file = str(saved_file)
        shape = np.load(file)

    #resize factor 
    shape[0] = shape[0]//resize_factor
    shape[1] = shape[1]//resize_factor

    return shape

