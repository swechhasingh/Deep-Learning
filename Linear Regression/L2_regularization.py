import numpy as np
import matplotlib.pyplot as plt 

N=50
X = np.linspace(0,10,N)
Y = 0.5*X + np.random.randn(N)
X = np.vstack([np.ones(N), X]).T
Y[-1] += 30
Y[-2] += 30

plt.scatter(X[:,1], Y)

#maximize likelihood solution
w = np.linalg.solve(X.T.dot(X),X.T.dot(Y))
Yhat = X.dot(w)

plt.plot(X[:,1], Yhat, label='max likelihood')

#MAP solution(L2 regularization)
L2 = 1000.0
w = np.linalg.solve(L2*np.eye(2) + X.T.dot(X),X.T.dot(Y))
YMap = X.dot(w)

plt.plot(X[:,1], YMap, label='MAP')
plt.legend()
plt.show()
