from __future__ import division
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import optimizers

def get_class_labels(dir):
    return os.listdir(dir)

def get_class_images(classes, dir):
    class_paths = []
    
    for label in classes:
        image_paths = np.array([])
        class_path = os.path.join(dir, label)
        images = os.listdir(class_path)
        for image in images:
            image_path = os.path.join(class_path, image)
            image_paths = np.append(image_paths, image_path)
        class_paths.append(image_paths)
    return class_paths

dir = "animals/animals"
print(get_class_images(get_class_labels(dir), dir))