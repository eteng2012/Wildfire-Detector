
import os
import datetime

import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

print(tf.version.VERSION)

#Reads in the training/validation data, standardizes test images for training

TRAINING_DIR = "/Users/ericteng/Desktop/Wildfire_Project/FIRE-SMOKE-DATASET/Train"
training_datagen = ImageDataGenerator(rescale=1./255,
zoom_range=0.15,
horizontal_flip=True,
fill_mode='nearest')

VALIDATION_DIR = "/Users/ericteng/Desktop/Wildfire_Project/FIRE-SMOKE-DATASET/Test"
validation_datagen = ImageDataGenerator(rescale = 1./255)

#Settings for training/validation image generators, batch size and shuffling of images

train_generator = training_datagen.flow_from_directory(
TRAINING_DIR,
target_size=(224,224),
shuffle = True,
class_mode='categorical',
batch_size = 90)

validation_generator = validation_datagen.flow_from_directory(
VALIDATION_DIR,
target_size=(224,224),
class_mode='categorical',
shuffle = True,
batch_size= 10)

from keras.applications.inception_v3 import InceptionV3
from keras_preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout

#Sets the input size of the training model, initalizes the base model as InceptionV3

input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

#Adds additional layers at the top of the model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#train only the added top layers first
for layer in base_model.layers:
  layer.trainable = False

#rmsprop is the optimizer I found to work best for InceptionV3 compatibility
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# Tried SGD, less effective
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

model.summary()

#Keeps logs of training

log_dir = "/Users/ericteng/Desktop/Wildfire_Project/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Quarter images used in first training phase, rest in second: 
#Test: 1800/4 images = 90 images/step * 5 steps/epoch * 1 epoch
#Train: 200/4 images = 10 images/step * 5 steps/epoch * 1 epoch

#First training phase with only additional top layers

history = model.fit(
train_generator,
steps_per_epoch = 5,
epochs = 1,
validation_data = validation_generator,
validation_steps = 5,
callbacks=[tensorboard_callback])

#Second training phase trains the top two InceptionV3 layers together with the additional top layers
#This is done to make the two separate models more compatible when combined
for layer in model.layers[:249]:
  layer.trainable = False
for layer in model.layers[249:]:
  layer.trainable = True

#Recompiles the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

#Second training phase with the top two InceptionV3 layers and additional top layers together
history = model.fit(
train_generator,
steps_per_epoch = 5,
epochs = 3,
validation_data = validation_generator,
validation_steps = 5,
callbacks=[tensorboard_callback])

#Saves model to file to be loaded for testing
model.save('/Users/ericteng/Desktop/Wildfire_Project/fire_detection_model/inceptionV3_model.keras')

#Evaluate model
loss, acc = model.evaluate(validation_generator, verbose=2)

"""
Below are results: 
5/5 ━━━━━━━━━━━━━━━━━━━━ 21s 3s/step - acc: 0.5269 - loss: 18.8395 - val_acc: 0.6000 - val_loss: 1.5888
Epoch 1/3
5/5 ━━━━━━━━━━━━━━━━━━━━ 43s 7s/step - acc: 0.5888 - loss: 1.8642 - val_acc: 0.9000 - val_loss: 0.8842
Epoch 2/3
5/5 ━━━━━━━━━━━━━━━━━━━━ 13s 3s/step - acc: 0.8931 - loss: 0.2433 - val_acc: 1.0000 - val_loss: 3.4648e-04
Epoch 3/3
5/5 ━━━━━━━━━━━━━━━━━━━━ 14s 2s/step - acc: 0.9859 - loss: 0.0437 - val_acc: 0.9800 - val_loss: 0.1411
20/20 - 2s - 94ms/step - acc: 0.9700 - loss: 0.1467

If train entire InceptionV3 by itself:
5/5 ━━━━━━━━━━━━━━━━━━━━ 259s 44s/step - acc: 0.5946 - loss: 0.8693 - val_acc: 0.5000 - val_loss: 1156.3503
Epoch 2/2
5/5 ━━━━━━━━━━━━━━━━━━━━ 148s 27s/step - acc: 0.8473 - loss: 0.3203 - val_acc: 0.5600 - val_loss: 102.9655
20/20 - 11s - 550ms/step - acc: 0.5000 - loss: 113.5597
"""