import numpy as np
import cv2
import subprocess
import os
import random
import tensorflow as tf
os.chdir("C:\\Users\\biber\\Desktop\\CAT_VS_DOG")

class Cat_Dog():
    def __init__(self):
        self.inp, self.out = self.get_inp_out()
        self.model = self.get_model()
    def read_process_data(self, directory, output):
        data = []
        t=0
        for i in os.listdir(directory):
            img = cv2.imread(directory + "\\" + i, 0)
            img = cv2.resize(img, (28,28))
            data.append([img / 255, output])
            data.append([cv2.rotate(img,cv2.ROTATE_180) / 255, output])
            data.append([cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) / 255, output])
            data.append([cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE) / 255, output])
        print(len(data))
        return data
    def shuffle(self, data):
        random.shuffle(data)
        return data
    def get_inp_out(self):
        data_cat = self.read_process_data("cats", 0)
        data_dog = self.read_process_data("dogs", 1)
        data = data_cat + data_dog
        data = self.shuffle(data)
        inp, out = [], []
        for i in data:
            inp.append(i[0])
            out.append(i[1])
        return np.asarray(inp), np.asarray(out)
    def get_model(self):
        model = tf.keras.Sequential([ 
            tf.keras.layers.Conv2D(32, (3, 3), padding = "same", activation = "relu", input_shape = (28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), padding = "same", activation = "relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding = "same", activation = "relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = "relu"),     
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                    optimizer = 'adam',
                    metrics = ["accuracy"])
        return model
    def call(self):
        self.model.fit(self.inp, self.out, epochs = 40)
if __name__ == "__main__":
    C_D = Cat_Dog()
    C_D.call()
    