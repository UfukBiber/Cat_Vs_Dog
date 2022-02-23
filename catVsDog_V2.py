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
    image_size = (224, 224),
    validation_split = 0.2,
    subset = "training",
    seed = 1234,
    shuffle = True,
    batch_size = 16
)
valData = tf.keras.utils.image_dataset_from_directory(
    "Images",
    image_size = (224, 224),
    validation_split = 0.2,
    subset = "validation",
    seed = 1234,
    shuffle = True,
    batch_size = 16
)


baseModel = tf.keras.applications.vgg16.VGG16(input_shape = (224, 224, 3),
                                            include_top = False,
                                            weights = "imagenet")
for layer in baseModel.layers:
    layer.trainable = False
x = layers.Dropout(0.25)(baseModel.output)
x = layers.Flatten()(x)
x = layers.Dense(256, activation = "relu")(x)
x = layers.Dropout(0.5),
x = layers.Dense(1, activation = "sigmoid")(x)
model = tf.keras.models.Model(baseModel.input, x)


model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(trainData, epochs = 3, validation_data = valData)
