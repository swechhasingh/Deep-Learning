import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

X = []
Y = []
R = pd.read_csv("data_1d.csv", header=None).as_matrix()
X = R[:,0]
Y = R[:,1]
X = np.array(X)
Y = np.array(Y)
N = len(X)

print X
print Y

plt.title('1D Linear Regression')
plt.scatter(X,Y)
plt.axis('equal')
plt.xlabel('X data')
plt.ylabel('Y data')

#find value of a and b in equation y = a*x + b
Xmu = np.sum(X)/N   #Xmu = X.mean()
XSqMu = np.sum(np.square(X))/N  #XSqMu = X.dot(X)/N
denominator = XSqMu - Xmu*Xmu
Ymu = Y.mean()

a = (X.dot(Y)/N - Xmu*Ymu)/denominator
b = (Ymu*XSqMu - Xmu*X.dot(Y)/N)/denominator

#print "a = ",a, "b =", b

Yhat = a*X + b
plt.plot(X,Yhat)
plt.axis('equal')
plt.show()

#calculate R-squared
n = Y - Yhat
d = Y - Y.mean()
R2 = 1 - n.dot(n)/d.dot(d)
print "R-squared : ", R2


