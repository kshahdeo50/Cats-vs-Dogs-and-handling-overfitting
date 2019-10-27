# import basic librarys and dependencies
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Dropout,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

'''Load data
Begin by downloading the dataset. This tutorial uses a filtered version of Dogs vs Cats dataset from Kaggle.
Download thtrain_dir = os.path.join(path,'train')
validation_dir = os.path.join(path,'validation')e archive version of the dataset and store it in the "/tmp/" directory.'''

path_zip = tf.keras.utils.get_file('cats_and_dogs.zip',origin = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",extract = True)

path = os.path.join(os.path.dirname(path_zip),'cats_and_dogs_filltered')


#After extracting its contents, assign variables with the proper file path for the training and validation set.
train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')
validation_cats_dir = os.path.join(validation_dir,'cats')
validation_dogs_dir = os.path.join(validation_dir,'dogs')

batch_size = 128
epochs = 15
IMG_HEIGHT = 100
IMG_WIDTH = 100

'''Data preparation
Format the images into appropriately pre-processed floating point tensors before feeding to the network:

Read images from the disk.
Decode contents of these images and convert it into proper grid format as per their RGB content.
Convert them into floating point tensors.
Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.'''


train_image_generator = ImageDataGenerator(rescale = 1./255)
validation_image_generator = ImageDataGenerator(rescale = 1./255)


train_set = train_image_generator.flow_from_directory(
        'cats_and_dogs_filtered/train',
        target_size=(IMG_HEIGHT,IMG_WIDTH),
        batch_size=128,
        class_mode='binary')

test_set = validation_image_generator.flow_from_directory(
                'cats_and_dogs_filtered/validation',
                 target_size = (IMG_HEIGHT,IMG_WIDTH),
                 batch_size =128,
                 class_mode =  'binary')
                 
#Visualize the images data
sample_images_tr, _ = next(train_set)

def plotImages(images_arr):
    fig, axes = plt.subplots(1,5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    tight_layout()
    plt.show()
    
 plt.plotImages(sample_images_tr[:5])
 
 #create the model
 
 model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

#Compile the model

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
#Train the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#Visualize training results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
    
