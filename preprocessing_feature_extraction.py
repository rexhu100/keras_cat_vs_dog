import os
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import reduce

conv_base = VGG16(
    include_top=False,
    input_shape=(150, 150, 3),
    weights="imagenet"
)


def _extract_features(img_generator, batch_count):
    input_list = []
    label_list = []
    i = 0
    for input_batch, label_batch in img_generator:
        if i == batch_count:
            break

        input_list.append(conv_base.predict(input_batch))
        label_list.append(label_batch)

        i += 1

    features = reduce(lambda x, y: np.append(x, y, 0), input_list)
    label = reduce(lambda x, y: np.append(x, y, 0), label_list)

    return features, label


def get_validation_data():
    base_dir = os.path.join("data", "subsample")
    validation_dir = os.path.join(base_dir, "validation")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary"
    )

    return _extract_features(validation_generator, 50)


def get_training_data():
    base_dir = os.path.join("data", "subsample")
    train_dir = os.path.join(base_dir, "train")

    print(conv_base.summary())

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary"
    )

    return _extract_features(train_generator, 100)
