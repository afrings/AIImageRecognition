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
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers, layers
from tensorflow.keras.optimizers import RMSprop

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

pre_trained_model = InceptionV3(input_shape = (255, 255, 3),
                                include_top = False,
                                weights = None)
local_weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer("mixed10")
last_output = last_layer.output
x = layers.Dense(1024, activation='relu')(last_output)
x = layers.Dropout(0.2)(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(1, activation='softmax')(x)
model = Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

