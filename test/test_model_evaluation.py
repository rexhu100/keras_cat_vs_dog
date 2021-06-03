import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

base_dir = os.path.join("..", "data", "subsample")
test_dir = os.path.join(base_dir, "test")

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)

model = tf.keras.models.load_model("../small_convnet_v2.h5")

result = model.evaluate(
    test_generator,
    return_dict=True
)

print(result)

score_datagen = ImageDataGenerator(rescale=1.0/255)
score_generator = score_datagen.flow_from_directory(
    "../data/submission_data",
    target_size=(150, 150),
    batch_size=20,
    class_mode=None,  # don't generate labels
    shuffle=False  # don't shuffle
)


print(score_generator.class_indices)
print(test_generator.class_indices)

pred = (model.predict(score_generator) > 0.5).astype("int32")
print(pred[:10])
print([os.path.splitext(x[6:]) for x in score_generator.filenames[:10]])

submission = pd.DataFrame({
    "id": [int(os.path.splitext(x[6:])[0]) for x in score_generator.filenames],
    "label": [x[0] for x in pred]
})

submission.to_csv("../data/submission_data/submission.csv")