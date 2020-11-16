
# coding: utf-8

# ###  Train baseline model

#train model
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping

#load data

batch_size = 32
img_height= 224
img_width= 224

train_ds = image_dataset_from_directory(
  r'GoldStandard_tvt/train',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = image_dataset_from_directory(
  r'GoldStandard_tvt/validation',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#check class names and tensor shapes
class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# dataset performance configuration
AUTOTUNE = AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# create resnet model w imgnet weights
resnet_model = ResNet50(weights='imagenet', 
                        include_top=False, 
                        input_shape=(224, 224, 3),
                        pooling= None)
resnet_model.summary()

model= Sequential()

model.add(Rescaling(1./255, input_shape=(img_height, img_width, 3)))

for layer in resnet_model.layers:
    layer.trainable = True

model.add(resnet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))


model.summary()

#config model
opt = keras.optimizers.Adam(learning_rate=0.0001) #smaller lr for initialized weights
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

#callback
early_stopping= [EarlyStopping(monitor='val_acc', patience=50)]

#train

history= model.fit(
        train_ds,
        epochs=100, 
        validation_data=val_ds,
        callbacks= early_stopping
        )

print('cnn traininng done, save model')

#safe model
model.save('resnet_model1_finetuned.h5')

#test data
test_ds = image_dataset_from_directory(
  r'GoldStandard_tvt/test',
  shuffle=False,
  image_size=(img_height, img_width),
  batch_size=batch_size)



#test model

y_pred = np.around(model.predict(test_ds)).astype(int)

np.save('predictions.npy', y_pred)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_ds, batch_size=32)
print("test loss, test acc:", results)
print(results)



acc_hist =np.asarray(history.history['acc'])
np.save('acc_history.npy', acc_hist)

val_acc= np.asarray(history.history['val_acc'])
np.save('val_acc_history.npy', val_acc)

loss_hist =np.asarray(history.history['loss'])
np.save('loss_history.npy', loss_hist)

val_loss= np.asarray(history.history['val_loss'])
np.save('val_loss_history.npy', val_loss)

print('Script finished successfully')