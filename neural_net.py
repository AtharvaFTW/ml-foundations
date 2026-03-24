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

def sgd(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """
    Gradient descent update
    """
    W1 = W1 - lr * dW1
    W2 = W2 - lr * dW2
    b1 = b1 - lr * db1
    b2 = b2 - lr * db2

    return W1, b1, W2, b2

def momentum_update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr,v, beta =0.9):
    #update velocity for each parameter
    v["W1"] = beta * v["W1"] + (1-beta) *dW1
    v["b1"] = beta * v["b1"] + (1-beta) *db1
    v["W2"] = beta * v["W2"] + (1-beta) *dW2
    v["b2"] = beta * v["b2"] + (1-beta) *db2

    W1 = W1 - lr * v['W1']
    b1 = b1 - lr * v['b1']
    W2 = W2 - lr * v['W2']
    b2 = b2 - lr * v['b2']

    return W1, b1, W2, b2, v

def adam_update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr, m, v, t, beta1=0.9, beta2=0.999 , eps= 1e-8):
    #update m (momentum)
    m["W1"] = beta1 * m["W1"] + (1-beta1) * dW1
    m["b1"] = beta1 * m["b1"] + (1-beta1) * db1
    m["W2"] = beta1 * m["W2"] + (1-beta1) * dW2
    m["b2"] = beta1 * m["b2"] + (1-beta1) * db2

    #update v (velocity)
    v["W1"] = beta2 * v["W1"] + (1-beta2) * dW1**2
    v["b1"] = beta2 * v["b1"] + (1-beta2) * db1**2
    v["W2"] = beta2 * v["W2"] + (1-beta2) * dW2**2
    v["b2"] = beta2 * v["b2"] + (1-beta2) * db2**2

    #bias correction
    m_corrected_W1 = m["W1"] / (1-beta1**t)
    m_corrected_b1 = m["b1"] / (1-beta1**t)
    m_corrected_W2 = m["W2"] / (1-beta1**t)
    m_corrected_b2 = m["b2"] / (1-beta1**t)

    v_corrected_W1 = v["W1"] / (1-beta2**t)
    v_corrected_b1 = v["b1"] / (1-beta2**t)
    v_corrected_W2 = v["W2"] / (1-beta2**t)
    v_corrected_b2 = v["b2"] / (1-beta2**t)

    #update weights

    W1 = W1 - lr * m_corrected_W1 / (np.sqrt(v_corrected_W1) + eps)
    b1 = b1 - lr * m_corrected_b1 / (np.sqrt(v_corrected_b1) + eps)
    W2 = W2 - lr * m_corrected_W2 / (np.sqrt(v_corrected_W2) + eps)
    b2 = b2 - lr * m_corrected_b2 / (np.sqrt(v_corrected_b2) + eps)

    return W1, b1, W2, b2, m, v

def train(optimizer):
    # Training Data
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])

    #Hyperparameters 
    epochs = 300
    losses = []
    if optimizer == "sgd":
        lr = 0.1
    elif optimizer == "momentum":
        lr = 0.1
    elif optimizer == "adam":
        lr = 0.01
    
    W1, b1, W2, b2 = init_params()

    v = {"W1": np.zeros_like(W1) , "b1":np.zeros_like(b1),
         "W2": np.zeros_like(W2) , "b2":np.zeros_like(b2)}
    
    m = {"W1": np.zeros_like(W1) , "b1":np.zeros_like(b1),
         "W2": np.zeros_like(W2) , "b2":np.zeros_like(b2)}
    t = 0
    
    print(f"Training for {optimizer}")
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            act1 , act2 = forward(x,W1,b1,W2,b2)
            loss = compute_loss(act2, y)
            total_loss += loss
            dW1, db1, dW2, db2 = backward(x,y,act1,act2,W1,b1,W2)
            if optimizer == "sgd":
                W1, b1, W2, b2 = sgd(W1,b1,W2,b2,dW1,db1,dW2,db2,lr)
            elif optimizer == "momentum":
                W1, b1, W2, b2, v = momentum_update(W1,b1,W2,b2,dW1,db1,dW2,db2,lr,v)
            elif optimizer == "adam":
                t+=1
                W1, b1, W2, b2, m, v = adam_update(W1,b1,W2,b2,dW1,db1,dW2,db2,lr,m,v,t)
        losses.append(total_loss/4)
    
        if epoch%50 == 0:
            print(f"Epoch {epoch}, Loss:{total_loss/4:.4f}")
    
    print("\n")
    return losses

if __name__ =="__main__":
    print("\n")
    for opt in ["sgd", "momentum" , "adam"]:
        losses = train(opt)
        plt.plot(losses, label = opt)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("XOR Training Loss - Optimizer Comparision")
    plt.legend()
    plt.savefig("images/optimizer_comparison.png")
    print("Saved optimizer_comparison.png")
