from tensorflow.keras import layers
import pandas as pd
import os, shutil
import tensorflow as tf
import cv2

###################### Use only if images are mixed#####################
def divideData():
    path = "cat_dog"
    try:
        os.mkdir("Images/Cats")
        os.mkdir("Images/Dogs")
    except:
        pass
    labels = pd.read_csv("cat_dog.csv")
    for imagePath in os.listdir(path):
        imPath = os.path.join(path, imagePath)   
        label = labels.loc[labels["image"] == imagePath]["labels"].values
        if label == 1:
            shutil.move(imPath, "Images/Dogs")
        else:
            shutil.move(imPath, "Images/Cats")
    os.rmdir("cat_dog")
###########################################################################



trainData = tf.keras.utils.image_dataset_from_directory(
    "Images",
    image_size = (28, 28),
    validation_split = 0.2,
    subset = "training",
    seed = 1234,
    shuffle = True,
    batch_size = 32
)
valData = tf.keras.utils.image_dataset_from_directory(
    "Images",
    image_size = (28, 28),
    validation_split = 0.2,
    subset = "validation",
    seed = 1234,
    shuffle = True,
    batch_size = 32
)


model = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.Rescaling(1./255, input_shape=(28, 28, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.BatchNormalization(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation = "sigmoid")
])


model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(trainData, epochs = 5, validation_data = valData)
