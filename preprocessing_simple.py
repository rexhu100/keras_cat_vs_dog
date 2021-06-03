import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_images():
    base_dir = "data/subsample"

    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")

    # For simple CNN model
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150,),
        batch_size=20,
        class_mode="binary"
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150,),
        batch_size=20,
        class_mode="binary"
    )

    # Testing the generators
    for data_batch, labels_batch in train_generator:
        print(f"Data batch shape (training): {data_batch.shape}")
        print(f"Label batch shape (training): {labels_batch.shape}")
        break

    for data_batch, labels_batch in validation_generator:
        print(f"Data batch shape (validation): {data_batch.shape}")
        print(f"Label batch shape (validation): {labels_batch.shape}")
        break

    return train_generator, validation_generator


