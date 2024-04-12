import numpy as np

class Sigmoid:
    def funct(self, x):
        return(1/(1 + np.exp(-x)))

    def funct_prime(self, x):
        return(self.funct(x)*(1-self.funct(x)))

class Tanh:
    def funct(self, x):
        return np.tanh(x);

    def funct_prime(self, x):
        return 1- self.funct(x)**2;

class Relu:
    def funct(self, x):
        return np.maximum(0, x) 

    def funct_prime(self, x):
        return np.where(x >= 0, 1, 0) 

