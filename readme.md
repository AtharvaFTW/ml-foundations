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
- A derivative is just the slope at a point
- Gradients are how neural networks know which direction to update

**Files:** `gradient.py`

---

## Day 3 - Loss Functions from Scratch

Implemented MSE, Binary Cross-Entropy, and Categorical Cross-Entropy in NumPy without any ML libraries.

**Key Concepts**
- Cross entropy derived from information theory
- Loss is just -log(probability assigned to correct class)
- BCE punishes confident wrong answers far harder than MSE
- Numerical stability via np.clip to prevent log(0)

**Files:** `loss_function.py`