#train model
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import base_CNN
from data import load_data, data_as_npy_arrays
import random as random
import sklearn.metrics as skm
from tensorflow.keras.applications import VGG16




# #load data

#shape = np.load(r'savedshape.npy')
#img_height= int(shape[0]/4)
#img_width= int(shape[1]/4)

#train_generator = generate_data(data_path=r'E:\Babette\MasterThesis\GoldStandard_tvt_sub\train')
#validation_generator= generate_data(data_path=r'E:\Babette\MasterThesis\GoldStandard_tvt_sub\validation')

train_data= data_as_npy_arrays(data_path=r'E:\Babette\MasterThesis\GoldStandard_tvt_sub\train', height=224, width=224, name='train')
validation_data= data_as_npy_arrays(data_path=r'E:\Babette\MasterThesis\GoldStandard_tvt_sub\validation', height=224, width=224, name='val')


x_train, y_train= load_data(array=True ,dp=r'train_data.npy', n='train', h=224, w=224)
x_val, y_val= load_data(array=True ,dp=r'val_data.npy', n='val', h=224, w=224)



#Initiate Model

model = VGG16(
    include_top=True,
    weights= None , #"imagenet",
    input_tensor=None,
    input_shape=(300,300,3),
    pooling=None,
    classes=1,
   classifier_activation="sigmoid",
)


model= Sequential()

for layers in vgg16.layers[:-1]:
    model.add(layers)
    
#for layers in model.layers[:-1]:
#    layers.trainable = False

model.add(Dense(1, activation='sigmoid'))

#model.summary()


#config model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


#parameters
early_stopping= [EarlyStopping(monitor='val_acc', patience=10)]


history = model.fit(
        x_train,
        y_train,
        epochs=150,
        callbacks=early_stopping,
        batch_size = 32,
        validation_data=(x_val, y_val),
        #steps_per_epoch=63,
        #verbose=1,  # Logs once per epoch.
        )



model.save('VGG16_CNN.h5')

#test data

test_data= data_as_npy_arrays(data_path=r'test', height=224, width=224, name='test')

x_test, y_test= load_data(array=True ,dp=r'test_data.npy', n='test', h=224, w=224)

#test model
y_pred = np.around(model.predict(x_test)).astype(int)
np.save('predictions.npy', y_pred)


report = skm.classification_report(y_test, y_pred )
print(report) 

#confusion Matrix
cm = skm.confusion_matrix(y_test, y_pred)
print(cm)


acc_hist =np.asarray(history.history['acc'])
np.save('acc_history.npy', acc_hist)

val_acc= np.asarray(history.history['val_acc'])
np.save('val_acc_history.npy', val_acc)



loss_hist =np.asarray(history.history['loss'])
np.save('loss_history.npy', loss_hist)

val_loss= np.asarray(history.history['val_loss'])
np.save('val_loss_history.npy', val_loss)
