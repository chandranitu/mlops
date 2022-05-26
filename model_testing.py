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

#pip3 install scikit-image

print(tf.__version__)

# Explore Data
data_dir = pathlib.Path("/home/hadoop123/data_ml/seg_train")
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)

#ls -alrt models
model_loaded = tf.keras.models.load_model('/home/hadoop123/data_ml/models/')
model_loaded.summary()
from PIL import Image
import numpy as np
from skimage import transform
def process(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (150, 150, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image
pred_label=model_loaded.predict(process('/home/hadoop123/data_ml/splash.jpeg'))
print(class_names[np.argmax(pred_label)])
pred_label

#!zip -r models.zip models/
