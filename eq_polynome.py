"""
Created on Thu Oct  8 17:37:51 2025

@author: Jules Malavieille
"""

import numpy as np
import matplotlib.pyplot as plt 
# from param_value import val_param_3eq

# size = 10
# [alpha, r, B, P, c] = val_param_3eq(size)

# Ex bon paramètres : 
alpha = 0.2
r = 10
B = 3
P = 10
c = 3


def equation_eq(N):
    Y = []
    N_sol = [0]
    for i in range(len(N)):
        y = -alpha*N[i]**3 + r*N[i]**2 + N[i]*(-alpha*c**2 - B*P) + r*c**2
        Y.append(y)
        
        if Y[i-1] < 0 and Y[i] > 0:
            N_sol.append((N[i-1]+ N[i])/2)
        if Y[i-1] > 0 and Y[i] < 0:
            N_sol.append((N[i-1]+ N[i])/2)
        if Y[i] == 0:
            N_sol.append(N[i])   
            
    return Y, N_sol


N_min = 0
N_max = 100
nb_val = 1000

N = np.linspace(N_min, N_max, nb_val)
Y, N_sol = equation_eq(N)

plt.plot(N, Y, color="blue")
plt.plot([N_min, N_max], [0, 0],"black")
plt.plot([0,0], [-10000, 10000], color="blue")
plt.plot(N_sol, [0,0], ".", color="red", label="Equilibres", linewidth=10)
plt.xlabel("Valeur de N")
plt.ylabel("Y du polynome")
plt.title("Dynamique du système pour ∆>0")
plt.grid()
plt.legend()

print(N_sol)


