from tensorflow.keras import layers
import tensorflow as tf 


VALIDATION_SPLIT = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(
    "TrainImages",
    batch_size = 32,
    image_size = (160, 160),
    validation_split = VALIDATION_SPLIT,
    subset = "training",
    shuffle = True,
    seed = 1234
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "TrainImages",
    batch_size = 32,
    image_size = (160, 160),
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

Inputs = layers.Input(shape = (160, 160, 3))
x = data_augmentation(Inputs)
x = layers.Rescaling(1. / 255)(x)
for filterSize in [32, 64, 128, 256, 256]:
    residual = x
    x = layers.Conv2D(filters=filterSize, kernel_size = (3, 3), activation = "relu", padding = "same")(x)
    x = layers.Conv2D(filters=filterSize, kernel_size = (3, 3), activation = "relu", padding = "same")(x)
    x = layers.MaxPooling2D()(x)
    residual = layers.Conv2D(filterSize, kernel_size = 1, strides = 2, padding = "same")(residual)
    x = x + residual

x = layers.Flatten()(x)
x = layers.Dense(1, activation = "sigmoid")(x)



model = tf.keras.models.Model(Inputs, x)

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(train_ds, validation_data = val_ds, epochs = 50, 
        callbacks = [tf.keras.callbacks.ModelCheckpoint("ConvModel_With_Augmentation_Normalization", save_best_only = True, monitor = "val_accuracy")])



test_ds = tf.keras.utils.image_dataset_from_directory(
    "TestImages",
    batch_size = 32,
    image_size = (180, 180)
)

model.evaluate(test_ds)