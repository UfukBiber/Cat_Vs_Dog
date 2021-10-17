import Dense
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
train_images = train_images.reshape(train_images.shape[0],-1).T


dense_1 = Dense.Dense(10,train_images)
c = dense_1.call()
dense_2 = Dense.Dense(1, c)
d = dense_2.call()
print(d)
