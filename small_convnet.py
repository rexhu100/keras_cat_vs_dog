import tensorflow as tf
from tensorflow.keras import models, layers


def build_small_convnet() -> models.Sequential:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3,), activation="relu", input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2,)))
    model.add(layers.Conv2D(64, (3, 3,), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2,)))
    model.add(layers.Conv2D(128, (3, 3,), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2,)))
    model.add(layers.Conv2D(128, (3, 3,), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2,)))
    model.add(layers.Dropout(0.5))  # Dropout layer for regularization

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))

    print(model.summary())

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.binary_crossentropy,
        metrics=["acc"]
    )

    return model
