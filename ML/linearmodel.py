import numpy as np
from scipy.optimize import minimize
from ML.lossfunction import MSE, MSE_grad

def LinearRegression(x, y):
	initial_theta=[100,100]
	theta = minimize(fun = MSE, 
                     x0 = initial_theta, 
                     args = (x, y),
                     method = 'BFGS',
                     jac = MSE_grad)
	return np.array(theta.x).reshape(-1,1)



