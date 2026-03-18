import numpy as np
import matplotlib.pyplot as plt
import math


def f_single(x: int) -> int:
    """
    f(x) = x². Takes a number, returns a number.
    """
    return x**2

def f_double(x: int , y: int) -> int:
    """
    f(x,y)= x² + y². Takes two numbers, returns a number
    """
    return x**2 + y**2

def numerical_gradient_single(x: int, h: int = 0.0001) -> int:
    """
    Compute the numerical gradient of f_single at point x.
    Definition: (f(x+h) - f(x))/h 
    """
    return (f_single(x+h) - f_single(x))/h

def numerical_gradient_double(x: int,y: int, h: int=0.0001) -> int:
    """
    Compute the partial derivaties of f_double at point(x,y)
    Returns TWO values: gradient w.r.t x, gradient w.r.t y.
    Remember:   Partial w.r.t x means freeze y.
                Partial w.r.t y means freeze x.
    """
    grad_x = (f_double(x+h, y)- f_double(x,y))/h
    grad_y = (f_double(x, y+h)- f_double(x,y))/h
    return grad_x, grad_y

def f_combined(x: int) -> int:
    """
    Compute using the chain rule
    """
    return (f_single(x) * 3)


def plot_gradient():
    """
    Plot f(x) = x² and its numerical gradient on the same graph.
    x should range from -5 to 5
    """
    x = np.linspace(-5,5,100)
    y = f_single(x)
    gradients = [numerical_gradient_single(xi) for xi in x]
    

    plt.plot(x,y, label="f(x)=x²")
    plt.plot(x,gradients, label = "gradient")
    plt.legend()
    plt.show()


if __name__ =="__main__":
    print("\n")
    print(f"f_single output on 3: {f_single(3)}")
    print(f"f_single output on 5: {f_single(5)}")
    print(f"f_double output on (2,3): {f_double(2,3)}")
    print(f"Direct gradient using f_combine on 2 {(f_combined(2+0.0001)- f_combined(2))/0.0001}")
    print(f"Chain rule gradient {numerical_gradient_single(2) * 3}")
    print(f"Does the chain_rule match the direct_gradient? : {math.isclose((f_combined(2+0.0001)- f_combined(2))/0.0001 , numerical_gradient_single(2) * 3 )}")
    print(f"numerical_gradient_single output on 3: {numerical_gradient_single(3)}")
    print(f"numerical_gradient_single output on 5: {numerical_gradient_single(5)}")
    print(f"numerical_gradient_double output on (2,3): {numerical_gradient_double(2,3)}")
    plot_gradient()
    print("\n")

