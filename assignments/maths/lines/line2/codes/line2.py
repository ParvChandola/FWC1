import numpy as np

n = np.array([4,3])
c = 12
d = 4
e1 = np.array([1,0])
n1 = n[0]*n[0] + n[1]*n[1]
norm_n = np.sqrt(n1)

x1 = (d*norm_n+c)/n.T@e1
x2 = (-d*norm_n+c)/n.T@e1

print(f"The two points are ({x1,0}) and ({x2,0})")
