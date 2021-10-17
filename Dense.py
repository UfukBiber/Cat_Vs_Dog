import numpy as np
class Dense:
    def __init__(self, nodes, Input):
        self.input = Input
        self.input_size = Input.shape
        self.nodes = nodes
        self.w, self.b = self.init_weight_and_bias()
    def Relu(self, x):
        return np.where(x <= 0, 0, x)
    def init_weight_and_bias(self):
        weigth = np.random.rand(self.input_size[0],1,self.nodes)
        bias = 0.00
        return weigth, bias
    
    def call(self):
        A = np.dot(self.w.T,self.input) + self.b
        A = np.squeeze(A)
        A = self.Relu(A)
        return A





