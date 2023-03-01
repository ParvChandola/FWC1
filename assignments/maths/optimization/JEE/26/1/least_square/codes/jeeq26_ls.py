import numpy as np
import matplotlib.pyplot as plt
import math
import cvxpy as cv
import matplotlib
import subprocess
import shlex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#line parameters
O = np.array([2,1,-1])
A = np.array([1,1,0])
m = np.array([2,-1,1])

#creating the variable
Lambda = cv.Variable()

#optimization problem
obj = cv.Minimize(cv.norm(A+(Lambda*m)-O))
prob = cv.Problem(obj, [])
prob.solve()

#solution
print("The minimum distance is : ", obj.value)
print("The minimum value of lambda is : ", Lambda.value)