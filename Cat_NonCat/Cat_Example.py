import numpy as np
from tensorflow.keras.datasets import fashion_mnist



class Dense_Layer():
    def __init__(self, X, Y):
        self.X = self.ModifyData(X)
        self.Y = Y / 10.0
        self.m = self.X.shape[1]
        self.w = np.zeros((self.X.shape[0],1),dtype=float)
        self.b = 0.00
    def ModifyData(self, x):
        x = x.reshape(x.shape[0],-1).T 
        x = x / 255.0
        return x
    def sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))
    def propagate(self):
        A = self.call()
        cost = self.calculateCost(A)
        cost = np.squeeze(np.array(cost))
        grads = self.calculateGrads(A)
        return cost, grads 
    def calculateGrads(self, A):
        dw = (1 / self.m) * (np.dot(self.X, (A-self.Y).T))
        db = (1 / self.m) * (np.sum(A-self.Y))
        return {'dw' : dw, 'db' : db}
    def calculateCost(self, A):
        cost = (-1 / self.m) * (np.dot(self.Y,np.log(A).T) + np.dot(1-self.Y, np.log(1-A).T))
        return cost
    def optimize(self, learning_rate):
        cost, grads = self.propagate()
        self.w -= grads['dw'] * learning_rate
        self.b -= grads['db'] * learning_rate
        return cost
    def call(self):
        result = self.sigmoid(np.dot(self.w.T,self.X)+self.b)
        return result

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(train_labels.shape)
    dense_layer = Dense_Layer(train_images,train_labels)
    for i in range(10):
        cost = dense_layer.optimize(10)
        print(cost)



