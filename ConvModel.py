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



Inputs = layers.Input(shape = (180, 180, 3))
x = layers.Rescaling(1. / 255)(Inputs)

x = layers.Conv2D(filters=32, kernel_size = (3, 3), activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Conv2D(filters=32, kernel_size = (3, 3), activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Conv2D(filters=32, kernel_size = (3, 3), activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Conv2D(filters=32, kernel_size = (3, 3), activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Conv2D(filters=32, kernel_size = (3, 3), activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(1, activation = "sigmoid")(x)


model = tf.keras.models.Model(Inputs, x)
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
print(model.summary())




model.fit(train_ds, validation_data = val_ds, epochs = 100, 
        callbacks = [tf.keras.callbacks.ModelCheckpoint("ConvModel", save_best_only = True, monitor = "val_accuracy"),
                    tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 3)])


test_ds = tf.keras.utils.image_dataset_from_directory(
    "TestImages",
    batch_size = 32,
    image_size = (180, 180)
)

model.evaluate(test_ds)