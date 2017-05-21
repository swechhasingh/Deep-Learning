import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#load data from
X = []
Y = []
R = pd.read_csv('data_2d.csv', header=None).as_matrix()
X = R[:,[0,1]]
X = np.c_[X,np.ones(100)]
Y = R[:,2]

X = np.array(X)
Y = np.array(Y)

#plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
plt.show()

#solve for weights "W"
A = (X.T).dot(X)
B = (X.T).dot(Y)
W = np.linalg.solve(A,B)
print "Weights W = ", W

#predicted data
Yhat = X.dot(W)

#plot the predicted data in red color 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],Yhat,c='red')
plt.show()

#calculate R-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
R2 = 1 - d1.dot(d1)/d2.dot(d2)
print "R-squared = ", R2














