from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def top_layer_model():
    model = Sequential();
    # model.add(layers.Input(shape=))
    model.add(layers.Flatten(input_shape=(4, 4, 512)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["acc"]
    )

    return model
