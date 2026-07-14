import numpy as np

def hypothesis(X, w):
    return X @ w

def compute_cost(X, y, w):
    m = len(y)
    errors = hypothesis(X, w) - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)    
    return cost

def compute_gradient(X, y, w):
    m = len(y)
    gradient = (1 / m) * X.T @ (hypothesis(X, w) - y)
    return gradient

def gradient_descent(X, y, w, learning_rate, iterations):
    
    cost_history = []
    
    for i in range(iterations):
        
        gradient = compute_gradient(X, y, w)
        w = w - learning_rate * gradient
        cost_history.append(compute_cost(X, y, w))
        
    return w, cost_history