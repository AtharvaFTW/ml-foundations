import numpy as np


def mse_loss(y_true: np.ndarray , y_pred: np.ndarray) -> float:
    """
    Takes true and predicted values and returns the mean squared error loss.
    """
    error = y_true - y_pred
    mse = (error**2).mean()
    return mse


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Takes true and predicted values and returns the binary_crossentropy.
    """
    y_pred = np.clip(y_pred, 1e-15,1-1e-15)
    bce = (-(y_true*np.log(y_pred) + (1- y_true)*np.log(1-y_pred))).mean()
    return bce

def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Takes true and predicted values and returns the categorical_crossentropy.
    """
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    cce = -np.sum(y_true * np.log(y_pred), axis = 1).mean()
    return cce

if __name__ =="__main__":

    #MSE
    y_true = np.array([1.0,2.0,3.0])
    y_pred = np.array([1.5,2.5,3.5])
    print(f"MSE: {mse_loss(y_true, y_pred)}")

    #Binary CE
    y_true = np.array([1,0,1])
    y_pred = np.array([0.9,0.1,0.8])
    print(f"BCE: {binary_crossentropy(y_true , y_pred)}")

    #Categorical CE
    y_true = np.array([[0,1,0] ,[1,0,0]])
    y_pred = np.array([[0.1,0.7,0.2],[0.8,0.1,0.1]])
    print(f"CCE: {categorical_crossentropy(y_true, y_pred)}")

    #Plotting 

    import matplotlib.pyplot as plt

    p = np.linspace(0.01,0.99, 200)

    bce_curve = -(np.log(p))
    mse_curve = (1-p)**2

    plt.figure(figsize=(8,5))
    plt.plot(p,bce_curve, label = "Binary CE (y_true= 1)")
    plt.plot(p,mse_curve, label = "MSE (y_true= 1)")
    plt.xlabel("y_pred")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

