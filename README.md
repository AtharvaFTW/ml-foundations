# ML Foundations

Building ML from scratch - no shortcuts, no black boxes.

---

## Day 1 - Matrix Multiplication from Scratch

Implemented matrix multiplication using nested loops in pure Python with no NumPy.
Includes dimension mismatch validation.

**Key Concepts**
- How matrix multiplication actually works under the hood.
- Every deep learning operation is matrix multiplication at its core.

**Files:** `matrix_operations.py`

---

## Day 2 - Numerical Gradients

Computed numerical gradient of f(x)=x² and f(x,y)=x²+y² using the definition (f(x+h)-f(x))/h.
Verified chain rule by hand.

**Key concepts:**
- A derivative is just the slope at a point.
- Gradients are how neural networks know which direction to update.

**Files:** `gradient.py`

---

## Day 3 - Loss Functions from Scratch

Implemented MSE, Binary Cross-Entropy, and Categorical Cross-Entropy in NumPy without any ML libraries.

**Key Concepts**
- Cross entropy derived from information theory.
- Loss is just -log(probability assigned to correct class).
- BCE punishes confident wrong answers far harder than MSE.
- Numerical stability via np.clip to prevent log(0).

**Files:** `loss_function.py`

## Day 4 - Value Class from Scratch (Andrej Karpathy)

Implemented __add__, __mul__, backward() and tanh() from ground up.

**Key Concepts**
- Autograd tracks every operation in a computation graph so gradients can flow backwards automatically
- The chain rule is how gradients move through the graph - each operation multiplies the incoming gradient by its local derivative
- Topological sort ensures gradients are computed in the right order - output before inputs
- Leaf nodes (like `a` and `b`) accumulate gradients via `+=` because they can be used in multiple operations
- tanh saturates for large inputs - gradient approaches 0, which is why weight initialization matters

**Files:** `micrograd.py`

## Day 5 - Built 2-Layer neural network in pure NumPy.

Implemented init_params, sigmoid, forward, compute_loss, backward, update_params, training functions

**Key Concepts**
`init_params` - Initialized random weights 
`sigmoid` - A function for sigmoid activation
`forward` - The forward functionality of the network
`compute_loss` - Computes the loss
`backward` - Backpropagation
`update_params` - This updates the current weights to minimize loss
`train` - The function that encapsulates everything we built so far and trains the neural network

XOR Results table -

| Input | Target | Predicted |
|-------|--------|-----------|
| [0,0] | 0      | 0.0301    |
| [0,1] | 1      | 0.9369    |
| [1,0] | 1      | 0.9491    |
| [1,1] | 0      | 0.0613    |

**Key Insight:** A linear model cannot solve XOR. The hidden layer learns an intermediate representation that makes XOR linearly separable, that's why depth matters.

Loss curve image - 

![XOR Training Loss](images/XOR_Training_Loss.png)

**Files:** `neural_net.py`
