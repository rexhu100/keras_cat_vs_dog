import os
import shutil


def create_subsample():
    original_dataset_dir = "data/train"

    base_dir = "data/subsample"
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")
    test_dir = os.path.join(base_dir, "test")

    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    train_cat_dir = os.path.join(train_dir, "cats")
    validation_cat_dir = os.path.join(validation_dir, "cats")
    test_cat_dir = os.path.join(test_dir, "cats")

    train_dog_dir = os.path.join(train_dir, "dogs")
    validation_dog_dir = os.path.join(validation_dir, "dogs")
    test_dog_dir = os.path.join(test_dir, "dogs")

    os.mkdir(train_cat_dir)
    os.mkdir(validation_cat_dir)
    os.mkdir(test_cat_dir)
    os.mkdir(train_dog_dir)
    os.mkdir(validation_dog_dir)
    os.mkdir(test_dog_dir)

    # Copy over cat images to subsample folder
    for fname in [f"cat.{i}.jpg" for i in range(1000)]:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cat_dir, fname)
        shutil.copyfile(src, dst)

    for fname in [f"cat.{i}.jpg" for i in range(1000, 1500)]:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cat_dir, fname)
        shutil.copyfile(src, dst)

    for fname in [f"cat.{i}.jpg" for i in range(1000, 1500)]:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cat_dir, fname)
        shutil.copyfile(src, dst)

    # Copy over dog images to subsample folder
    for fname in [f"dog.{i}.jpg" for i in range(1000)]:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dog_dir, fname)
        shutil.copyfile(src, dst)

    for fname in [f"dog.{i}.jpg" for i in range(1000, 1500)]:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dog_dir, fname)
        shutil.copyfile(src, dst)

    for fname in [f"dog.{i}.jpg" for i in range(1000, 1500)]:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dog_dir, fname)
        shutil.copyfile(src, dst)

    # Check the results
    print(f"Total training cat images: {len(os.listdir(train_cat_dir))}")
    print(f"Total validation cat images: {len(os.listdir(validation_cat_dir))}")
    print(f"Total test cat images: {len(os.listdir(test_cat_dir))}")

    print(f"Total training dog images: {len(os.listdir(train_dog_dir))}")
    print(f"Total validation dog images: {len(os.listdir(validation_dog_dir))}")
    print(f"Total test dog images: {len(os.listdir(test_dog_dir))}")
