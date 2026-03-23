import numpy as np

def init_params():
    """
    This function returns W1, b1, W2, b2.
    """
    W1 = np.random.randn(4,2) * 0.01
    W2 = np.random.randn(1,4) * 0.01
    b1 = np.zeros(4)
    b2 = np.zeros(1)

    return W1, b1, W2, b2

def sigmoid(x):
    return 1/ (1+ np.exp(-x))

def forward(x, W1, b1, W2, b2):
    """
    This function returns output and any intermediate values you'll need for backprop
    """ 
    a1 = np.tanh(np.matmul(W1,x) + b1)
    a2 = sigmoid(np.matmul(W2,a1) + b2)

    return a1,a2

def compute_loss(output, y):
    """
    Computes MSE Loss.
    """
    loss = np.mean((output-y)**2)
    return loss

def backward(x, y, a2, a1, output, W1, b1, W2, b2):
    """
    Returns gradients dW1, db1, dW2, db2.
    """
    n = y.size

    dL_da2 = (a2 - y)/n
    dz2 = dL_da2 * a2 * (1- a2)
    dW2 = np.outer(dz2, a1)
    db2 = dz2

    da1 = W2.T@ dz2
    dz1 = da1 * (1- np.tanh(W1 @ x + b1)**2)
    dW1 = np.outer(dz1, x)
    db1 = dz1

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """
    Gradient descent update
    """
    W1 = W1 - lr * dW1
    W2 = W2 - lr * dW2
    b1 = b1 - lr * db1
    b2 = b2 - lr * db2

    return W1, b1, W2, b2
