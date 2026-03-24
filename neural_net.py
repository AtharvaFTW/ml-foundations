import numpy as np
import matplotlib.pyplot as plt

def init_params():
    """
    This function returns W1, b1, W2, b2.
    """
    W1 = np.random.randn(4,2)
    W2 = np.random.randn(1,4)
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

def backward(x, y, a1, a2, W1, b1, W2):
    """
    Returns gradients dW1, db1, dW2, db2.
    """
    n = y.size

    dL_da2 = 2 * (a2 - y)/n
    dz2 = dL_da2 * a2 * (1- a2)
    dW2 = np.outer(dz2, a1)
    db2 = dz2

    da1 = W2.T@ dz2
    dz1 = da1 * (1-(a1)**2)
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

def train():
    # Training Data
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])

    #Hyperparameters 
    epochs = 120
    lr = 1.0
    losses = []
    W1, b1, W2, b2 = init_params()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            act1 , act2 = forward(x,W1,b1,W2,b2)
            loss = compute_loss(act2, y)
            total_loss += loss
            dW1, db1, dW2, db2 = backward(x,y,act1,act2,W1,b1,W2)
            W1, b1, W2, b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,lr)
        losses.append(total_loss/4)
    
        if epoch%20 == 0:
            print(f"Epoch {epoch}, Loss:{total_loss/4:.4f}")
        
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("XOR Training Loss")
    plt.savefig("xor_loss_curve.png")
    print("Loss curve saved.")



if __name__ =="__main__":
    print("\n")
    train()
