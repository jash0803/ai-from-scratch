import numpy as np
import time

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2

def gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def adagrad(learning_rate=0.1, epsilon=1e-8, iterations=500):
    start_time = time.time()
    x, y = -1.2, 1.0
    grad_accum = np.zeros(2)
    positions = [(x, y)]

    for epoch in range(iterations):
        grad = gradient(x, y)
        grad_accum += grad ** 2
        adjusted_grad = grad / (np.sqrt(grad_accum) + epsilon)
        x -= learning_rate * adjusted_grad[0]
        y -= learning_rate * adjusted_grad[1]
        positions.append((x, y))
        if np.linalg.norm(grad) < 1e-6:
            print(f"Adagrad Converged in {epoch + 1} epochs")
            break

    end_time = time.time()
    print(f"Adagrad Execution Time: {end_time - start_time:.4f} seconds")

    return positions, epoch + 1, end_time - start_time