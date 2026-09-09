# ANN for Regression — PyTorch

## 1. What is an ANN?

An **Artificial Neural Network (ANN)** is a neural network made of connected neurons that learns patterns from data by adjusting **weights and biases**.

For regression, the network predicts a **continuous numerical value**.

**Example:** Predict electrical energy output from power plant conditions.

---

## 2. Problem Used

**Dataset:** Power Plant dataset

**Input features:**

* `AT` → Ambient Temperature
* `V` → Exhaust Vacuum
* `AP` → Ambient Pressure
* `RH` → Relative Humidity

**Target:**

* `PE` → Electrical Energy Output

This is a **regression problem** because the output is a continuous number.

---

## 3. Basic Workflow

```text
Raw Data
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Convert to Tensors
   ↓
Dataset
   ↓
DataLoader
   ↓
ANN
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Updated Weights
   ↓
Repeat
```

---

## 4. Data Preparation

### Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

We use training data to learn and test data to evaluate performance.

### Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Important:**

* `fit_transform()` → only on training data
* `transform()` → on test data

This prevents **data leakage**.

---

## 5. Tensor, Dataset and DataLoader

### Tensor

PyTorch works primarily with **tensors**.

```python
torch.tensor(data, dtype=torch.float32)
```

### TensorDataset

Combines input and target:

```python
dataset = TensorDataset(X_tensor, y_tensor)
```

### DataLoader

Loads data in **mini-batches**:

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Remember

```text
Tensor      → stores data
Dataset     → organizes samples
DataLoader  → provides batches
```

---

## 6. ANN Architecture

The network used:

```text
Input Layer
    ↓
4 neurons
    ↓
6 neurons + ReLU
    ↓
6 neurons + ReLU
    ↓
1 neuron
    ↓
Output
```

Architecture:

```text
4 → 6 → 6 → 1
```

### Why 4 input neurons?

There are 4 input features:

```text
AT, V, AP, RH
```

### Why 1 output neuron?

The model predicts one value:

```text
PE
```

### Why ReLU?

ReLU introduces **non-linearity**, allowing the network to learn complex relationships.

```text
ReLU(x) = max(0, x)
```

### Why no activation on the output?

For ordinary regression, the output should be able to produce a continuous value, so we normally use **no output activation**.

---

## 7. Loss Function

For regression:

```python
criterion = nn.MSELoss()
```

**MSE — Mean Squared Error**

It measures the difference between predicted and actual values.

Lower MSE → better prediction.

---

## 8. Optimizer

```python
optimizer = optim.Adam(model.parameters())
```

**Adam** updates the model's weights and biases to reduce the loss.

---

## 9. Training Loop

The most important part:

```python
optimizer.zero_grad()

outputs = model(xb)

loss = criterion(outputs, yb)

loss.backward()

optimizer.step()
```

Remember:

```text
zero_grad()
    ↓
Forward Pass
    ↓
Calculate Loss
    ↓
Backward Pass
    ↓
Update Weights
```

### What does each step do?

| Code               | Purpose             |
| ------------------ | ------------------- |
| `zero_grad()`      | Clear old gradients |
| `model(xb)`        | Make prediction     |
| `criterion()`      | Calculate loss      |
| `loss.backward()`  | Calculate gradients |
| `optimizer.step()` | Update weights      |

---

## 10. Epoch

**Epoch = one complete pass through the entire training dataset.**

Example:

```python
for epoch in range(100):
```

The model trains for **100 epochs**.

---

## 11. Evaluation

During evaluation:

```python
model.eval()

with torch.no_grad():
    predictions = model(X_test)
```

### `model.eval()`

Puts the model into evaluation mode.

### `torch.no_grad()`

Prevents gradient calculation because we are not training.

---

## 12. Saving the Best Model

```python
torch.save(model.state_dict(), "best_model.pt")
```

Save the model when it achieves the **best validation/test performance**, rather than blindly keeping the final epoch.

---

# Interview Questions

### 1. What is ANN?

A neural network consisting of interconnected neurons that learns patterns by adjusting weights and biases.

### 2. Why do we scale input features?

To bring features to a similar scale and help the neural network train more effectively.

### 3. Why use `fit_transform()` only on training data?

To prevent **data leakage** from the test set.

### 4. Why use ReLU?

It introduces non-linearity and helps the network learn complex patterns.

### 5. Why no activation function in the regression output layer?

Because regression requires an unrestricted continuous output.

### 6. Why MSE for regression?

MSE measures the squared difference between predicted and actual values and is commonly used for continuous-value prediction.

### 7. What does `loss.backward()` do?

It calculates gradients of the loss with respect to the model parameters using backpropagation.

### 8. What does `optimizer.step()` do?

It updates the model's weights and biases using the calculated gradients.

### 9. Tensor vs Dataset vs DataLoader?

```text
Tensor     → data
Dataset    → samples
DataLoader → batches
```

### 10. What happens during training?

```text
Input
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradient
 ↓
Weight Update
```

---

# Quick Revision

```text
Problem        → Regression
Dataset        → Power Plant
Inputs         → AT, V, AP, RH
Output         → PE
Split          → 80% Train / 20% Test
Scaling        → StandardScaler
Architecture   → 4 → 6 → 6 → 1
Activation     → ReLU
Loss           → MSE
Optimizer      → Adam
Batch Size     → 32
Epochs         → 100
Training       → Forward → Loss → Backward → Update
Evaluation     → model.eval() + torch.no_grad()
```

## One-Line Interview Explanation

> "I built a PyTorch ANN regression model using the Power Plant dataset, standardized the input features, used a 4-6-6-1 architecture with ReLU hidden layers, trained it using MSE loss and Adam optimizer, and evaluated the model on unseen test data."
