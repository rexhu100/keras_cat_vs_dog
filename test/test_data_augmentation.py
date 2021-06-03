from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    base_dir = "../data/subsample"
    train_dir = os.path.join(base_dir, "train")
    train_cat_dir = os.path.join(train_dir, "cats")

    fnames = [os.path.join(train_cat_dir, fname) for fname in os.listdir(train_cat_dir)]
    img_path = fnames[2]

    img = image.load_img(img_path, target_size=(150, 150,))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    print(x.shape)

    # Use this to randomly transform the data
    datagen = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        print(batch)
        i += 1
        if i % 4 == 0:
            break

    plt.show()
