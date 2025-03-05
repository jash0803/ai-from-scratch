import numpy as np

np.random.seed(42)  
x = np.random.rand(10, 1)  
y = (x > 0.5).astype(int)

n_samples, n_features = x.shape

learning_rate = 0.01
n_iters = 100

weights = np.zeros((n_features, 1))
bias = 0.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(y_true, y_pred):
    epsilon = 1e-9  # To prevent log(0)
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def gradient_descent(x, y, weights, bias, learning_rate, n_iters):
    for _ in range(n_iters):
        linear_pred = np.dot(x, weights) + bias  
        y_pred = sigmoid(linear_pred)

        dw = (1/n_samples) * np.dot(x.T,(y - y_pred))
        db = (1/n_samples) * np.sum(y - y_pred) 

        weights -= learning_rate * dw
        bias -= learning_rate * db

        print(f"Weight: {weights} & Bias: {bias}")
        print(f"Loss: {loss(y, y_pred)}")
    return weights, bias

weights, bias = gradient_descent(x, y, weights, bias, learning_rate, n_iters)

linear_pred = np.dot(x, weights) + bias
y_pred = sigmoid(linear_pred)

final_loss = loss(y, y_pred)

print(f"Trained Weight: {weights}")
print(f"Trained Bias: {bias}")
print(f"Final Loss: {final_loss}")