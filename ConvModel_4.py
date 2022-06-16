from regex import W
from tensorflow.keras import layers
import tensorflow as tf 


VALIDATION_SPLIT = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(
    "TrainImages",
    batch_size = 32,
    image_size = (180, 180),
    validation_split = VALIDATION_SPLIT,
    subset = "training",
    shuffle = True,
    seed = 1234
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "TrainImages",
    batch_size = 32,
    image_size = (180, 180),
    validation_split = VALIDATION_SPLIT,
    subset = "validation",
    shuffle = True,
    seed = 1234
)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2)
])


convBase = tf.keras.applications.vgg16.VGG16(
    weights = "imagenet",
    include_top = False,
    input_shape = (180, 180, 3)
)

convBase.trainable = True

for layer in convBase.layers[:-2]:
    layer.trainable = False

Inputs = tf.keras.layers.Input(shape = (180, 180, 3))
x = tf.keras.applications.vgg16.preprocess_input(Inputs)
x = convBase(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation = "relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation = "sigmoid")(x)


model = tf.keras.models.Model(Inputs, x)
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


model.fit(train_ds, validation_data = val_ds, epochs = 3, 
        callbacks = [tf.keras.callbacks.ModelCheckpoint("ConvModel_with_ImageNet_FineTuning", save_best_only = True, monitor = "val_accuracy")])



test_ds = tf.keras.utils.image_dataset_from_directory(
    "TestImages",
    batch_size = 32,
    image_size = (180, 180)
)

model.evaluate(test_ds)