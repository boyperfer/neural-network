from utilities import forward_propagation, error, back_propagation
import numpy as np


class NeuralNetwork:
    def __init__(self, x_train=[], x_test=[], y_train=[], y_test=[], w1=[], w2=[], b1=None, b2=None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def train(self, funct, alpha = 0.01, epoch = 10):
        acc =[]
        for j in range(epoch):
            l =[]
            for i in range(len(self.x_train)):
                out = forward_propagation(funct, self.x_train[i], self.w1, self.w2, self.b1, self.b2)
                l.append((error(out, self.y_train[i])))
                self.w1, self.w2, self.b1, self.b2 = back_propagation(self.x_train[i],
                        self.y_train[i], self.w1, self.w2, self.b1, self.b2, alpha, funct)
            print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(self.x_train)))*100)
            acc.append((1-(sum(l)/len(self.x_train)))*100)
        return(acc, self.w1, self.w2)

        
    def predict(self, funct, i, j):
        out = forward_propagation(funct, self.x_test[i], self.w1, self.w2, self.b1, self.b2)
        acc = 1 - error(out, self.y_test[i])
        
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        print("predicted value ",j , " ", out) 

        print("true value ",j , " ", self.y_test[i])

        print("acc ", j, " ", acc * 100) 

        return out;



