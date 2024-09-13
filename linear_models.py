import numpy as np

class LinearRegression:
    def __init__(self, iterations, learning_rate, dim, reg_var, epsilon):
        self.iterations = iterations
        self.learning_rate =  learning_rate
        self.w = np.zeros(dim)
        self.b = 0
        self.reg_var = reg_var
        self.epsilon = epsilon
        self.mean = None
        self.std = None
        self.normalize = False
        
        
    def compute_cost(self, X, y):
        m = X.shape[0]
        f_wb = np.dot(X, self.w) + self.b
        err = f_wb - y
        
        unreg_cost = (1 / (2 * m)) * np.sum(np.square(err))
        reg_cost = (self.reg_var/m) * np.square(self.w)
        
        return unreg_cost + reg_cost
    
    
    def compute_gradient(self, X, y):
        m = X.shape[0]
        f_wb = np.dot(X, self.w) + self.b
        err = f_wb - y
        
        dj_dw = (1 / m) * np.dot(X.T, err)
        dj_db = (1 / m) * np.sum(err)
        
        dj_dw += (self.reg_var / m) * self.w
        
        return dj_dw, dj_db
    
    
    def gradient_descent(self, X, y):
        
        prev_cost = float('inf')
        cost_history = []
        
        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X, y)
            self.w = self.w - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db
            
            current_cost = self.compute_cost(X, y)
            cost_history.append(current_cost)
            
            if (abs(current_cost - prev_cost) <= self.epsilon):
                print(f"Convergence at iteration {i} with cost {current_cost}")
                break
            
            prev_cost = current_cost
            
            if (i % 100 == 0):
                print(f"Iteration {i}: Cost = {current_cost}")
        return cost_history
    
    
    def normalize_features(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std
    
    
    def fit(self, X, y, normalize=True):
        self.normalize = normalize
        
        if normalize:
            X = self.normalize_features(X)
        
        cost_history = self.gradient_descent(X, y)
        
        return cost_history
    
    def predict(self, X):
        if self.normalize and self.mean is not None and self.std is not None:
            X = (X - self.mean) / self.std
        
        return np.dot(X, self.w) + self.b