#from google.colab import drive
#drive.mount("/content/drive")

import os
import pathlib
import random
import numpy as np
import pandas as pd

import scipy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential


print(tf.__version__)

# Explore Data

# Walk through pizza_steak directory and list number of files
print("Train data: ")

for dirpath, dirnames, filenames in os.walk("/home/hadoop123/data_ml/seg_train"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
print("Test data: ")

for dirpath, dirnames, filenames in os.walk("/home/hadoop123/data_ml/seg_test"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
print("Prediction data: ")

for dirpath, dirnames, filenames in os.walk("/home/hadoop123/data_ml/seg_prod"):
 print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


#random image
def view_random_image(target_dir, target_class):
  # We will view image from here
  target_folder = target_dir + target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder+'/'+random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis('off');
  print(f"Image shape {img.shape}")
  return img

img = view_random_image(target_dir='/home/hadoop123/data_ml/seg_train/',
                  target_class='buildings')



#Get the class name programmatically
data_dir = pathlib.Path("/home/hadoop123/data_ml/seg_train")
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)


plt.figure(figsize=(20, 10))
for i in range(18):
  plt.subplot(3, 6, i+1)
  class_name = random.choice(class_names)
  img = view_random_image(target_dir='/home/hadoop123/data_ml/seg_train/',
                  target_class=class_name)



# Prepare Data Model

train_dir = "/home/hadoop123/data_ml/seg_train/"
test_dir = "/home/hadoop123/data_ml/seg_test/"


# Create augmented data generator instance
train_datagen = ImageDataGenerator(rescale=1/255.,
                                   rotation_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1/255.)


# Load data(data, label) from directory and turn them into batches

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(150,150),
                                               batch_size=32,
                                               class_mode='categorical')
test_data = val_datagen.flow_from_directory(test_dir,
                                           target_size=(150,150),
                                           batch_size=32,
                                           class_mode='categorical')


#Basic model Buildinig (CNN Classifier)

model = Sequential([
  Conv2D(16, 3, padding='same', activation='relu', input_shape=(150,150,3)),
  MaxPool2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPool2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(len(class_names), activation='softmax')
])

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()



history_1 = model.fit(train_data,
                    epochs=2,
                    batch_size=32,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))


model.evaluate(test_data)

pd.DataFrame(history_1.history)[['loss','val_loss']].plot()
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('Loss');



pd.DataFrame(history_1.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy');

model.save('/home/hadoop123/data_ml/models/', save_format='tf')

