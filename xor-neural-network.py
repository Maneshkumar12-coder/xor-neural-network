import numpy as np

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(1)

# Network: 2 -> 4 -> 1
w1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

w2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

loss_history = []

for epoch in range(epochs):

    # Forward propagation
    h = sigmoid(X @ w1 + b1)
    out = sigmoid(h @ w2 + b2)

    # Mean squared error
    error = y - out
    loss = np.mean(error ** 2)
    loss_history.append(loss)

    # Backpropagation
    d_out = error * dsigmoid(out)

    d_h = (d_out @ w2.T) * dsigmoid(h)

    # Parameter updates
    w2 += learning_rate * (h.T @ d_out)
    b2 += learning_rate * np.sum(d_out, axis=0, keepdims=True)

    w1 += learning_rate * (X.T @ d_h)
    b1 += learning_rate * np.sum(d_h, axis=0, keepdims=True)

print("Predictions:")
print(out.round(2))

print("\nFinal Loss:", loss_history[-1])