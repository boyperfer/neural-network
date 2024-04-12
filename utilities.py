import numpy as np

def generate_wt(x, y):
    l =[]
    for i in range(x * y):
        l.append(np.random.randn())
    return np.array(l).reshape(x, y)

def error(out, Y):
    s =(np.square(Y- out))
    s = np.sum(s)/2
    return(s)

def forward_propagation(funct, x, w1, w2, b1, b2):
    # hidden
    # print("size x", x.size)
    # print("size w1", w1.size)
    z1 = x.dot(w1) + b1# input from layer 1
    a1 = funct.funct(z1)# out put of layer 2

    # Output layer
    z2 = a1.dot(w2) + b2# input of out layer
    a2 = funct.funct(z2)# output of out layer
    return a2

def back_propagation(x, y, w1, w2, b1, b2, alpha, funct): 

    # hidden layer
    z1 = x.dot(w1) + b1  
    a1 = funct.funct(z1) 

    # Output layer
    z2 = a1.dot(w2) + b2 
    a2 = funct.funct(z2) 

    # sigma for output layer
    d2 = np.multiply((y-a2), funct.funct_prime(z2))

    # Gradient for bias output
    b2_adj = np.sum(d2, axis=0, keepdims=True)

    # sigma for hidden layers
    d1 = np.multiply(w2.dot(d2.transpose()).transpose(), funct.funct_prime(z1))

    # Gradient for bias hidden
    b1_adj = np.sum(d1, axis=0, keepdims=True)

    # Gradient for output layer 
    w1_adj = x.transpose().dot(d1)

    # Gradient for hidden layer 
    w2_adj = a1.transpose().dot(d2)

    # Updating weights 
    w1 = w1 + (alpha*(w1_adj))
    w2 = w2 + (alpha*(w2_adj))
    b1 = b1 - (alpha*(b1_adj))
    b2 = b2 - (alpha*(b2_adj))

    return(w1, w2, b1, b2)

