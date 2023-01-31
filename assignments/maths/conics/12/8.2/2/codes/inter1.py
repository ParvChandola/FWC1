#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sympy as sy

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/parv/CoordGeo')

#local imports
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex

#circle parameters
c1 = np.array([1,0])
c2 = np.array([0,0])
r = 1

#defining the function
def f(x):
    return 2*(1-(x-1)**2)**(1/2)

def g(x):
    return 2*(1-x**2)**(1/2)

x = sy.Symbol("x")
area = float(sy.integrate(f(x), (x, 0, 1/2))+sy.integrate(g(x), (x, 1/2, 1))) 
print(f"The area of the region is {area}")

##Generating the circle
x_circ1= circ_gen(c1,r)
x_circ2= circ_gen(c2,r)

#Plotting the circle
plt.plot(x_circ1[0,:],x_circ1[1,:])
plt.plot(x_circ2[0,:],x_circ2[1,:])
plt.fill_between(x_circ2[0],x_circ2[1],where= (0.5 <= x_circ2[0])&(x_circ2[0] <= 1 ), color = 'cyan')
plt.fill_between(x_circ1[0],x_circ1[1],where= (0 <= x_circ1[0])&(x_circ1[0] <= 0.539 ), color = 'cyan')
#Labeling the coordinates
tri_coords = np.vstack((c1,c2)).T
plt.scatter(tri_coords[0],tri_coords[1])

vert_labels = ['c1','c2']
for i, txt in enumerate(vert_labels):
    label = "{}({:.0f},{:.0f})".format(txt, tri_coords[0,i],tri_coords[1,i]) #Form label as A(x,y)
    plt.annotate(label, # this is the text
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
plt.savefig('/sdcard/Download/latexfiles/conics/figs/inter1.png')
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
plt.show()
