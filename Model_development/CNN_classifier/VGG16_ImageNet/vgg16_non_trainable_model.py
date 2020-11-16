
# coding: utf-8

# ###  Train baseline model

# train model
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping


# load data

batch_size = 32
img_height = 224
img_width = 224

train_ds = image_dataset_from_directory(
    # r'E:\Babette\MasterThesis\GoldStandard_tvt\train',
    r'GoldStandard_tvt/train',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = image_dataset_from_directory(
    r'GoldStandard_tvt/validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# check class names and tensor shapes
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

vgg16 = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
)


model = Sequential()

model.add(Rescaling(1./255, input_shape=(img_height, img_width, 3)))

for layers in vgg16.layers[1:-1]:
    model.add(layers)
model.add(Dense(1, activation='sigmoid'))

for layers in model.layers[1:-1]:
    layers.trainable = False

model.summary()

# config model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# callback
early_stopping = [EarlyStopping(monitor='val_acc', patience=10)]

# train

history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=early_stopping
)

print('cnn traininng done, save model')

# safe model
model.save('vgg16_model1_imgnet_weights.h5')


# test data
test_ds = image_dataset_from_directory(
    r'GoldStandard_tvt/test',
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# test model

y_pred = np.around(model.predict(test_ds)).astype(int)

np.save('predictions.npy', y_pred)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_ds, batch_size=32)
print("test loss, test acc:", results)
print(results)


acc_hist = np.asarray(history.history['acc'])
np.save('acc_history.npy', acc_hist)

val_acc = np.asarray(history.history['val_acc'])
np.save('val_acc_history.npy', val_acc)

loss_hist = np.asarray(history.history['loss'])
np.save('loss_history.npy', loss_hist)

val_loss = np.asarray(history.history['val_loss'])
np.save('val_loss_history.npy', val_loss)

print('Script finished successfully')
