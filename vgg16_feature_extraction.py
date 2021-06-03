from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def build_fine_tuned_model():
    conv_base = VGG16(
        include_top=False,
        input_shape=(150, 150, 3),
        weights="imagenet"
    )

    conv_base.trainable = True

    for layer in conv_base.layers:
        if not layer.name.startswith("block5"):
            layer.trainable = False

    model = Sequential()
    model.add(conv_base)

    model.add(layers.Flatten(input_shape=(4, 4, 512)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    print(model.summary())

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["acc"]
    )

    return model
