import numpy as np

def PreprocessingData(data):
	#delete all the row in the data that contains NaN value
	loc = np.argwhere(np.isnan(data))
	for row in loc[:,0]:
		data = np.delete(data, row, 0)
	return data

def FeatureLabelSpliting(data):
	#auto-split the data into feature matrix and label vector
	m, n = data.shape
	x = data[:,0:n-1]
	y = data[:,n-1:]
	return x, y