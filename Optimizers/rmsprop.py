import numpy as np
import time

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2

def gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def rmsprop(learning_rate=0.01, decay_rate=0.9, epsilon=1e-8, max_iterations=500):
    start_time = time.time()
    x, y = -1.2, 1.0
    grad_squared = np.zeros(2)
    positions = [(x, y)]
    losses = [rosenbrock(x, y)]

    for epoch in range(max_iterations):
        grad = gradient(x, y)
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * (grad ** 2)
        adjusted_grad = grad / (np.sqrt(grad_squared) + epsilon)
        x -= learning_rate * adjusted_grad[0]
        y -= learning_rate * adjusted_grad[1]
        positions.append((x, y))
        losses.append(rosenbrock(x, y))

        if np.linalg.norm(grad) < 1e-6:
            break

    end_time = time.time()
    return positions, losses, epoch + 1, end_time - start_time