import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical 
from neural_network import NeuralNetwork
from utilities import generate_wt
from activations import Tanh, Relu, Sigmoid
import random

options = ["Pre process", "quit"]
functions = ["sigmoid", "tanh", "relu"]

def print_menu(options):
    print("Select an option:")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    print("Enter the number corresponding to your choice:")

def print_function(options):
    print("functions:")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

def get_choice(options):
    while True:
        try:
            print_menu(options)
            choice = int(input())
            if 1 <= choice <= len(options):
                if choice == 1:
                    (x_train, y_train), (x_test, y_test) = mnist.load_data()
                    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
                    x_train = x_train.astype('float32')
                    x_train /= 255
                    y_train = to_categorical(y_train)
                    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
                    x_test = x_test.astype('float32')
                    x_test /= 255
                    y_test = to_categorical(y_test)

                    # hidden bias
                    b1 = np.zeros(20)

                    # output bias
                    b2 = np.zeros(10)

                    w1 = generate_wt(x_test[0].size, 20)
                    w2 = generate_wt(20, y_test[0].size)
                    network = NeuralNetwork(x_train, x_test, y_train, y_test, w1, w2, b1, b2)
                    print_function(functions)
                    funct_input = int(input())
                    if funct_input == 1:
                        funct = Sigmoid()
                    elif funct_input == 2:
                        funct = Tanh()
                    elif funct_input == 3:
                        funct = Relu()
                    else:
                        print("Invalid choice. Please enter a number within the range.")
                        return None
                    print("Please enter a number of epoch.")
                    epochs = int(input())
                    network.train(funct, 0.01, epochs)
                    for j in range(epochs):
                        i = random.randint(1, 10000)
                        network.predict(funct, i, j)

                if choice == 2:
                    return None

            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

get_choice(options)
