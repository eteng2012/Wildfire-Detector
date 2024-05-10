
import tensorflow

#import keras_preprocessing
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout

model = tensorflow.keras.models.load_model('/Users/ericteng/Desktop/Wildfire_Project/fire_detection_model/inceptionV3_model.keras')
#model = tensorflow.keras.models.load_model('/Users/ericteng/Desktop/Wildfire_Project/fire_detection_model/handmade_model.keras')

#Below uses image files from a different dataset to further validate the models
#Because the dataset is different, we expect accuracies to be slightly lower.
#However, the inceptionV3 model did about the same, while the handmade one did better
#Commented out code parts are the options if want to test for No Fire rather than Fire

path = '/Users/ericteng/Desktop/Wildfire_Project/fire-detection-master/Datasets1-2/Validation/fire'
#path = '/Users/ericteng/Desktop/Wildfire_Project/fire-detection-master/Datasets1-2/Validation/noFire'
import os
import numpy as np
from keras_preprocessing import image
dir_list = os.listdir(path)
nfire = 0
nnofire = 0
for f in dir_list:
    img = image.load_img(path+'/'+f, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255
    classes = model.predict(x)
    if classes[0][0] > classes[0][1]:
        nfire = nfire + 1
        print('fire')
    else:
        nnofire = nnofire + 1
        print('not fire')

print('fire: ' + str(nfire) + '; nonfire: ' + str(nnofire))
#print('nonfire: ' + str(nnofire) + '; fire: ' + str(nfire))

#result with fire data and inceptionV3_model is 91.81%: 157 fire, 14 no fire
#result with fire data and handmade_mode is 78.95%, 135 fire, 36 no fire
#result with no-fire data and inceptionV3_model is 98.53%: 67 no fire, 1 fire
#result with no-fire data and handmade_mode is 91.12%, 62 no fire, 6 fire



#Below is the code to link a Raspberry Pi camera feed to the model to get real-time accuracy
'''

import cv2
from PIL import Image

import picamera
from picamera.array import PiRGBArray

# camera = picamera.PiCamera()
# camera.resolution = (224, 224)
# rawCapture = PiRGBArray(camera, size=resolution)
video = cv2.VideoCapture(0)
while True:

    # camera.capture(rawCapture, format="bgr")
    # im = rawCapture.array

    _, frame = video.read()
    im = Image.fromarray(frame, 'RGB')

    cv2.imshow("Capturing", frame)

    #Resizing into 224x224 because we trained the model with this image size.
    im = im.resize((224,224))
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255
    probabilities = model.predict(img_array)[0]
    #Calling the predict method on model to predict 'fire' on the image
    prediction = np.argmax(probabilities)
    #if prediction is 0, which means there is fire in the frame.
    if prediction == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print(probabilities[prediction])
    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(3)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

'''