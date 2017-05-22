import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

X = []
Y = []
R = pd.read_csv('data_poly.csv', header=None).as_matrix()
X = R[:,0]
Y = R[:,1]

X = np.array(X)
Y = np.array(Y)

plt.scatter(X,Y)
plt.plot(sorted(X),sorted(Y))
plt.show()