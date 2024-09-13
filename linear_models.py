import numpy as np

class LinearRegression:
    def __init__(self, iterations, learning_rate, dim, reg_var, epsilon):
        self.iterations = iterations
        self.learning_rate =  learning_rate
        self.w = np.zeros(dim)
        self.b = 0
        self.reg_var = reg_var
        self.epsilon = epsilon
        
        
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
        
        