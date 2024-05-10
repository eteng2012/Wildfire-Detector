# Tensorflow Wildfire Detection System

## Description

This project is a program that trains two different machine learning models to identify images on a camera feed. These models classify images based on whether or not the image has a fire in it.

The first python file, fire_detect_handmade_model.py, is a program that trains a model with layers set by hand. The second python file, fire_detect_inceptionV3_model.py, is a program that trains a model using a combination of both the InceptionV3 model and my own model layers for formatting and further accuracy.

The third python file, fire_detect_model_test,py, is a program that allows testing of each model with a user's own dataset of images. It also has a commented out portion that allows for real-time accuracy reading based on a Raspberry Pi camera feed.

The fire_detection_model folder contains saved trained models for both the handmade and InceptionV3 model so users can test the models without having to retrain them themselves.

The .gitattributes file is just for compatibility with Git LFS, as the .keras model files in the fire_detection_model folder are too large for normal Github pushing.

## Requirements

To use the python files, one needs to install numpy, tensorflow, and keras onto their python environment. 

To use the Raspberry Pi functionality, more libraries will be needed, including cv2, PIL, and picamera. Please note that the libaries/functionality may be out of date for the newest versions of Raspberry Pi.

## Output

The two training files will output the model summary, epoch summary, and final accuracy based on validation data. The testing file will output the number of fire/non-fire images it sees in a given dataset.

## Support

If you have any questions or want to see the dataset used to train the models, please feel free to email me at eteng2012@gmail.com.

Thanks for stopping by!
