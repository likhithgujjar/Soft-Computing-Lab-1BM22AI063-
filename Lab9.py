# 9a) Single Layer Perceptron using Delta Rule (Gradient Descent)
import numpy as np

# Step function (activation)
def step(x):
    return 1 if x >= 0 else 0

# Training data: AND logic gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND output

# Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.1
epochs = 10

for epoch in range(epochs):
    total_error = 0
    for inputs, target in zip(X, y):
        net_input = np.dot(weights, inputs) + bias
        output = step(net_input)
        error = target - output
        # Delta rule weight update
        weights += learning_rate * error * inputs
        bias += learning_rate * error
        total_error += abs(error)
    print(f"Epoch {epoch+1}, Total Error: {total_error}")

print("Trained weights:", weights)
print("Trained bias:", bias)
