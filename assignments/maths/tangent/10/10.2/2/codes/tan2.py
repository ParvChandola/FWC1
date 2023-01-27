#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/parv/CoordGeo')

#local imports
from conics.funcs import circ_gen


#if using termux
import subprocess
import shlex

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

#initial values
r = 1
c = np.array([1.743,0])
theta_1 = (25*math.pi)/36
theta_2 = -(25*math.pi)/36


#calculations
P = c+r*np.array([math.cos(theta_1),math.sin(theta_1)])
Q = c+r*np.array([math.cos(theta_2),math.sin(theta_2)])
n1 = P-c
n2 = Q-c
m1 = np.array([1,-(n1[0]/n1[1])])
m2 = np.array([1,-(n2[0]/n2[1])])

m1_norm = np.linalg.norm(m1)
m2_norm = np.linalg.norm(m2)

theta = np.arccos((m1.T@m2)/(m1_norm*m2_norm))
theta_deg = theta * (180/math.pi)

print(f"The angle between the two tangents is {theta_deg} degrees")

T = np.array([0,0])

#Generating the circle
x_circ= circ_gen(c,r)

#generating the lines
x_TP = line_gen(T,P)
x_TQ = line_gen(T,Q)
x_QC = line_gen(Q,c)
x_PC = line_gen(P,c)

#plotting
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(x_TP[0,:],x_TP[1,:],label='$TP$')
plt.plot(x_TQ[0,:],x_TQ[1,:],label='$TQ$')
plt.plot(x_QC[0,:],x_QC[1,:],label='$QC$')
plt.plot(x_PC[0,:],x_PC[1,:],label='$PC$')

#Labeling the coordinates
tri_coords = np.vstack((T,P,Q,c)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['T','P','Q','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center                
plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#if using termux
#plt.savefig('../figs/problem1.pdf')
#subprocess.run(shlex.split("termux-open '../figs/problem1.pdf'")) 
plt.savefig('/sdcard/Download/latexfiles/tangent/figs/tan2.png')
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
plt.show()
















