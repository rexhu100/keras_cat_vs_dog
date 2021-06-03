import os

from preprocessing_simple import preprocess_images
from create_subsample import create_subsample
from util import plot_history


def main():
    if not os.path.exists("data/subsample"):
        create_subsample()

    # Version 1, 2: small CNN with/without dropout
    # from small_convnet import build_small_convnet
    #
    # train_generator, validation_generator = preprocess_images()
    # model = build_small_convnet()
    #
    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=100,
    #     epochs=100,
    #     validation_data=validation_generator,
    #     validation_steps=50
    # )
    #
    # model.save("small_convnet_v2.h5")

    # Version 3: Feature extraction with VGG16 model
    # from preprocessing_feature_extraction import get_training_data, get_validation_data
    # from top_layer_only import top_layer_model
    # train_input, train_label = get_training_data()
    # validation_input, validation_label = get_validation_data()
    #
    # model = top_layer_model()
    # history = model.fit(
    #     train_input, train_label,
    #     batch_size=20,
    #     epochs=30,
    #     validation_data=(validation_input, validation_label),
    #     # validation_steps=50
    # )

    # Version 4: Fine tuning
    from preprocessing_simple import preprocess_images
    from vgg16_feature_extraction import build_fine_tuned_model

    train_generator, validation_generator = preprocess_images()
    model = build_fine_tuned_model()

    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50
    )

    plot_history(history)


if __name__ == "__main__":
    main()
