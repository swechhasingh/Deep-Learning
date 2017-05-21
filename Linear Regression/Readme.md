# Linear Regression
Linear Regression is commonly known as **"line of best fit"**. It is a famous Supervised Learning algorithm for solving problems where there is linear relation between dependent variable(y) and independent variables(x) and the output(y) is a real number.

```
y = a*x + b  For every input(x) we have an output(y)
```

#### Examples
 1. One of the famous law of physics, Ohm's Law V= IR. If we compare this equation with equation of line 
 ```
 y = m*x+b, then y = V, x = I, m = R and b = 0.
 ```
In a electric circuit experiment, we can measure current(I) using ameter and voltage(V) across the battery but we don't       know the resistance of the load. So, we take N numuber of observations of (V,I) pair and plot these points on a 2D grid. We observe that these points form almost a perfect line.
 
 2. Moore's Law - According to Moore's Law, tranisitor count on integrated circuits doubles every two years.
 ```
 T(n) = (2^n)*T(0)  here,T(n) is transistor count in nth year.
 ```
This is an exponential equation but if take log of this equation we can convert it into a linear relation between log(T(n)) and n. Now, we can apply linear regression on this problem.

## What's included in this tutorial

1. Implementation of **1D Linear Regression** - linear_regression.py and data set for this - data_1d.csv
2. Implementation of **Moore's Law** - MooreLaw.py and data set for this - moore.csv

## How to run
Prerequisites python, numpy, matplotlib, pandas. Download Linear Regression folder.
Go inside the Linear Regression directory and run command.
For 1D Linear Regression-
```
python linear_regression.py
```
For Moore's Law - 
```
python MooreLaw.py
```



