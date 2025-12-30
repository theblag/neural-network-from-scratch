# Neural Network from Scratch

A neural network implementation built from scratch using NumPy, demonstrating the fundamentals of deep learning without relying on high-level frameworks like TensorFlow or PyTorch.

## Overview

This project implements a **feedforward neural network** (data flows forward through layers without loops) trained using **backpropagation** for binary classification:

- **Input Layer**: Accepts features from the dataset
- **Hidden Layer**: Configurable number of hidden units with sigmoid activation
- **Output Layer**: Single neuron with sigmoid activation for binary classification
- **Training Algorithm**: Backpropagation with batch gradient descent
- **Loss Function**: Binary cross-entropy (log loss)

## Features

- ✅ Custom neural network class built from scratch
- ✅ Sigmoid activation function
- ✅ Binary cross-entropy loss calculation
- ✅ Backpropagation implementation
- ✅ Gradient descent optimization
- ✅ Training and test loss evaluation
- ✅ Overfitting detection

## Usage

### 1. Load the Data

```python
import pandas as pd
df = pd.read_csv('data.csv')
```

### 2. Prepare Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Initialize and Train the Model

```python
# Initialize neural network
# n_features: number of input features
# hidden_units: number of neurons in hidden layer
customModel = NN(n_features=x_train.shape[1], hidden_units=5)

# Train the model
customModel.fit(x_train, y_train, epochs=500, loss_treshld=0.46)
```

### 4. Make Predictions

```python
y_pred = customModel.predict(x_test)
```

### 5. Evaluate Performance

```python
# Calculate losses
y_train_pred = customModel.predict(x_train)
train_loss = customModel.log_loss(y_train, y_train_pred)

y_test_pred = customModel.predict(x_test)
test_loss = customModel.log_loss(y_test, y_test_pred)

print(f"Training Loss: {train_loss}")
print(f"Test Loss: {test_loss}")
```

## How It Works

### Architecture

1. **Input Layer**: Takes in feature vector of size `n_features`
2. **Hidden Layer**: Contains `hidden_units` neurons with sigmoid activation
3. **Output Layer**: Single neuron with sigmoid activation (binary classification)

### Forward Propagation

```
z1 = X · W1 + b1
a1 = sigmoid(z1)
z2 = a1 · W2 + b2
y_pred = sigmoid(z2)
```

### Backpropagation

The network computes gradients using the chain rule:
- Output layer gradients: `dz2 = y_pred - y_true`
- Hidden layer gradients: `dz1 = (dz2 · W2) * a1 * (1 - a1)`
- Weight updates using gradient descent with learning rate = 0.5

### Loss Function

Binary cross-entropy loss:
```
L = -1/n * Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

## Key Parameters

- **n_features**: Number of input features (determined by dataset)
- **hidden_units**: Number of neurons in hidden layer (default: 5)
- **epochs**: Number of training iterations (default: 500)
- **learning_rate**: Step size for gradient descent (fixed at 0.5)
- **loss_treshld**: Loss threshold parameter (for potential early stopping)

## Neural Network Class Methods

- `__init__(n_features, hidden_units)`: Initialize network with random weights
- `sigmoid(x)`: Sigmoid activation function
- `log_loss(y_true, y_predicted)`: Calculate binary cross-entropy loss
- `fit(x, y, epochs, loss_treshld)`: Train the network
- `predict(x_test)`: Make predictions on new data
- `gradientDescent(x, y_true, epochs)`: Perform gradient descent optimization

## Model Evaluation

The notebook includes:
- Training vs Test loss comparison
- Overfitting detection mechanism
- Prediction comparison table (Real vs Predicted values)

## Learning Objectives

This implementation demonstrates:
- ✅ Manual weight initialization
- ✅ Forward propagation mechanics
- ✅ Backpropagation algorithm
- ✅ Gradient descent optimization
- ✅ Binary classification using neural networks
- ✅ Model evaluation techniques

## Status

This project was created for educational purposes to understand the inner workings of neural networks. More features are currently being developed.

## License

This project is open source and available for educational purposes.

## Author

Built as a learning project to understand the fundamentals of neural networks and backpropagation.

---