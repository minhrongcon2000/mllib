import numpy as np

def MSE(theta,x,y):
	#compute the L2 loss
	theta = np.array(theta).reshape(-1,1)
	m, n = x.shape[1], len(y) #m is the number of features and n is the number of labels
	X = np.concatenate((np.ones((n,1)),x),axis=1)
	h = np.dot(X,theta) #hypothesis vector
	return float(1.0/n * np.dot((h-y).T, h-y))

def MSE_grad(theta,x,y):
	#compute the partial derivative of L2 loss
	theta = np.array(theta).reshape(-1,1)
	m, n = x.shape[1], len(y) #m is the number of features and n is the number of labels
	X = np.concatenate((np.ones((n,1)),x),axis=1)
	h = np.dot(X,theta) #hypothesis vector
	return (2.0/n * np.dot(X.T, h-y)).flatten()