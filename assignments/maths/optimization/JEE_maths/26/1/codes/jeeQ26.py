import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import math
from cvxpy import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/parv/CoordGeo')

#local imports
from line.funcs import *
from triangle.funcs import *
from params import *
#if using termux
import subprocess
import shlex
#end if

#Line parameters
x1 = np.array([1,1,0])
ax1 = x1.reshape(3,-1)
x2 = np.array([2,1,-1])
ax2 = x2.reshape(3,-1)
m1 = np.array([2,-1,1])
am1 = m1.reshape(3,-1)
m2 = np.array([2,-1,1])
am2 = m2.reshape(3,-1)

M = np.block([[m1],[m2]]).T
# Create two scalar optimization variables.
lamda = Variable((2,1))
#Parametric equation of 2 lines
#A = ax1 + lamda1*am1
#B = ax2 + lamda2*am2

# Form objective function
obj = Minimize(norm(M@lamda + (ax2-ax1)))

# Form and solve problem.
prob = Problem(obj, [])
prob.solve()

#Optimal points on each line
A = x1-lamda.value[0]*m1
B = x2+lamda.value[1]*m2

print("Lambda values = ", -lamda.value[0], lamda.value[1])
print("Distance=", obj.value)
print("Point on line 1",A) 
print("Point on line 2",B)

#Points for plotting Line1
line1_point1 = x1 -1*(m1)
line1_point2 = x1 +5*(m1)
line1 = line_gen(line1_point1, line1_point2) 
#Points for plotting Line2
line2_point1 = x2 -1*(m2)
line2_point2 = x2 +5*(m2)
line2 = line_gen(line2_point1, line2_point2) 
#Points for plotting shortest distance between Line 1 and 2 
lineAB = line_gen(A, B) 

#Plotting lines
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(line1[0,:],line1[1,:], line1[2,:],label = 'Line1')
ax.plot3D(line2[0,:],line2[1,:], line2[2,:],label = 'Line2')
ax.plot3D(lineAB[0,:],lineAB[1,:], lineAB[2,:],label = 'AB')

ax.text(A[0],A[1],A[2],'A')
ax.text(B[0],B[1],B[2],'B')
plt.legend(['L1','L2','Normal'])

ax.view_init(60,30)
plt.grid()
plt.tight_layout()
plt.savefig('/sdcard/Download/latexfiles/optimization/figs/jee_q26.png')
plt.show()
