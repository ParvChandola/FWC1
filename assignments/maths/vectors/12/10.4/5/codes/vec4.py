import sympy as sym
import numpy as np

A = np.array([[2,0], [0,2], [6,-27]])
b = np.array([27,6,0])

x = 2 #no. of variables
AB = np.concatenate((A,b[:,None]),axis=1)

r1 = np.linalg.matrix_rank(A)
r2 = np.linalg.matrix_rank(AB)

if r1==r2==x:
    A_pinv = np.linalg.pinv(A)
    x = A_pinv@b
    print("The value of \mu and \lambda respectively are")
    print(x)
else:
    print("no solution")


