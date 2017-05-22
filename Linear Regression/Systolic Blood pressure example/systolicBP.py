import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 

data = pd.read_excel('mlr02.xls')
print data.columns     #print column names

X = data.as_matrix()

plt.scatter(X[:,1],X[:,0])
plt.show()

plt.scatter(X[:,2],X[:,0])
plt.show()

#Here Y = systolic blood pressure (i.e 'X1' 1st column)
#X = [X2,X3,1] - X2 is age in years and X3 is weigth in pounds
data['ones'] = 1
Y = data['X1']
X = data[['X2','X3','ones']]
X2 = data[['X2','ones']]
X3 = data[['X3','ones']]

#function to calculate R-squared
def cal_R2(X,Y):
	W = np.linalg.solve(X.T.dot(X),X.T.dot(Y))
	Yhat = X.dot(W)

	d1 = Y - Yhat
	d2 = Y - Y.mean()
	R2 = 1 - d1.dot(d1)/d2.dot(d2)
	return R2

print "R2 for X2 only : ", cal_R2(X2,Y)
print "R2 for X3 only : ", cal_R2(X3,Y)
print "R2 for X : ", cal_R2(X,Y)

#Try adding a new dimension that is just pure noise - use np.random.randn()
noise = np.random.randn(11)     # no of samples = 11
data['noise'] = noise
X = np.c_[X,noise]           #for adding column vector of noise to X use np.c_[X,noise]
print "R2 for X with noise : ", cal_R2(X,Y)



