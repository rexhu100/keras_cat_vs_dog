import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16

conv_base = VGG16(
    include_top=False,
    input_shape=(150, 150, 3),
    weights="imagenet"
)

