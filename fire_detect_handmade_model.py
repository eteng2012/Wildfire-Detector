import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#Reads in the training/validation data, standardizes test images for training

TRAINING_DIR = "/Users/ericteng/Desktop/Wildfire_Project/FIRE-SMOKE-DATASET/Train"
training_datagen = ImageDataGenerator(rescale=1./255,
zoom_range=0.15,
horizontal_flip=True,
fill_mode='nearest')

VALIDATION_DIR = "/Users/ericteng/Desktop/Wildfire_Project/FIRE-SMOKE-DATASET/Test"
validation_datagen = ImageDataGenerator(rescale = 1./255)

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

#Define model layers
										   
from tensorflow.keras.optimizers import Adam
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(224, 224, 3)),
tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
tf.keras.layers.Conv2D(256, (5,5), activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
tf.keras.layers.Conv2D(384, (5,5), activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(2048, activation='relu'),
tf.keras.layers.Dropout(0.25),
tf.keras.layers.Dense(1024, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(2, activation='softmax')])
model.compile(loss='categorical_crossentropy',
optimizer=Adam(learning_rate=0.0001),
metrics=['acc'])

#Train model on training data

history = model.fit(
train_generator,
steps_per_epoch = 10,
epochs = 2,
validation_data = validation_generator,
validation_steps = 10
)

#Saves model to file to be loaded for testing

model.save('/Users/ericteng/Desktop/Wildfire_Project/fire_detection_model/handmade_model.keras')

#Evaluate model
loss, acc = model.evaluate(validation_generator, verbose=2)

"""
Below are results:
10/10 ━━━━━━━━━━━━━━━━━━━━ 25s 1s/step - acc: 0.6162 - loss: 0.6617 - val_acc: 0.7700 - val_loss: 0.5614
Epoch 2/2
10/10 ━━━━━━━━━━━━━━━━━━━━ 7s 713ms/step - acc: 0.7730 - loss: 0.4892 - val_acc: 0.8100 - val_loss: 0.4620
20/20 - 0s - 20ms/step - acc: 0.7950 - loss: 0.5096
"""