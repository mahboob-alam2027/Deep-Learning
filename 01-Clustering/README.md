# Neuron Implementation — PyTorch

## 1. What is a Neuron?

A **neuron** takes inputs, multiplies them by weights, adds a bias, and produces an output.

### Formula

```text
z = x₁w₁ + x₂w₂ + x₃w₃ + b
```

Where:

* `x` → input
* `w` → weight
* `b` → bias
* `z` → output

---

## 2. PyTorch Implementation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

inputs = torch.tensor([1.0, 2.0, 3.0])

neuron = nn.Linear(
    in_features=3,
    out_features=1
)

output = neuron(inputs)
```

### What does `nn.Linear()` mean?

```python
nn.Linear(in_features=3, out_features=1)
```

```text
3 inputs → 1 neuron → 1 output
```

PyTorch automatically creates:

* 3 weights
* 1 bias

---

## 3. Understanding the Calculation

The neuron performs:

```text
Output = Weighted Sum + Bias
```

In PyTorch:

```python
neuron.weight @ inputs + neuron.bias
```

`@` performs matrix/vector multiplication.

This gives the same result as:

```python
output = neuron(inputs)
```

---

## 4. Tensor Dimensions

Remember:

```text
0D → Scalar
1D → Vector
2D → Matrix
3D+ → Tensor
```

Example:

```python
torch.tensor([1.0, 2.0, 3.0])
```

is a **1D tensor (vector)**.

---

# Interview Questions

### 1. What is a neuron?

A neuron computes a weighted sum of inputs, adds a bias, and produces an output.

### 2. What does `nn.Linear()` represent?

It represents a **fully connected linear layer** that performs:

```text
y = xWᵀ + b
```

### 3. What does `in_features=3` mean?

The layer receives **3 input features**.

### 4. What does `out_features=1` mean?

The layer contains **1 output neuron**.

### 5. What are weights and bias?

* **Weights** determine the importance of each input.
* **Bias** shifts the output.

### 6. How many parameters does this neuron have?

For 3 inputs:

```text
Weights = 3
Bias    = 1
Total   = 4 parameters
```

### 7. Why use `torch.manual_seed(42)`?

It makes random initialization reproducible, so the same initial weights can be generated again.

---

# Quick Revision

```text
Neuron
  ↓
Inputs × Weights
  ↓
Weighted Sum
  ↓
+ Bias
  ↓
Output
```

### PyTorch

```python
nn.Linear(3, 1)
```

means:

```text
3 inputs → 1 neuron → 1 output
```

### Most Important Formula

```text
z = Σ(xᵢwᵢ) + b
```
