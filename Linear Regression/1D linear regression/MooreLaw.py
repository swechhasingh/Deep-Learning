import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import re

T=[]     #transistor count    T(n) = (2^n)A 
Y=[]     #years
non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
	r = line.split('\t')
	x = int(non_decimal.sub('',r[1].split('[')[0]))
	y = int(non_decimal.sub('',r[2].split('[')[0]))
	T.append(x)
	Y.append(y)

T = np.array(T)
Y = np.array(Y)

#plot T on y axis and Y on x axis
plt.scatter(Y,T)
plt.xlabel('years')   #Y on x axis
plt.ylabel('Transistor count')  #T on y axis
plt.show()
 
#linear relation log(T(n)) = n*log(2) + log(A)   (T = a*Y + b)
T = np.log(T)
denominator = Y.dot(Y) - Y.mean()*Y.sum()
a = (Y.dot(T) - T.mean()*Y.sum())/denominator
b = (T.mean()*Y.dot(Y) - Y.dot(T)*Y.mean())/denominator

print "a = ", a, " b = ", b

#log plot of transistor count 
plt.scatter(Y,T)
That = a*Y + b
plt.plot(Y,That)
plt.show()

#Calculate R-squared to know how good our predicted model is
x1 = T - That
x2 = T - T.mean()
R2 = 1 - x1.dot(x1)/x2.dot(x2)
print "R-squared = ", R2

#Calculate the no of years it take to double transistor count
# log(transistorcount) = a*year + b
# transistorcount = exp(b) * exp(a*year)
# 2*transistorcount = 2 * exp(b) * exp(a*year) = exp(ln(2)) * exp(b) * exp(a * year) = exp(b) * exp(a * year + ln(2))
# a*year2 = a*year1 + ln2
# year2 = year1 + ln2/a
#Y2 = Y1 + ln2/a
print "time to double:", np.log(2)/a, "years"









