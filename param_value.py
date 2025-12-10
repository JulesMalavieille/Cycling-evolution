"""
Created on Wed Oct  8 16:03:33 2025

@author: Jules Malavieille
"""

import numpy as np
import matplotlib.pyplot as plt 
import sympy as sy

alpha = 10
r = 10
B = 10
P = 10
c = 8.9


def val_param_3eq(size):
    A = np.linspace(0.1, 10, size)
    R = np.linspace(0.1, 5, size)
    B = np.linspace(0.1, 10, size)
    P = np.linspace(0.1, 20, size)
    C = np.linspace(0.1, 10, size)
    D = np.zeros([len(A), len(R), len(B), len(P), len(C)])
    for i in range(len(A)):
        for j in range(len(R)):
            for k in range(len(B)):
                for l in range(len(P)):
                    for m in range(len(C)):
                        delta = -4*(C[m]**2 + B[k]*P[l]/A[i] - R[j]**2/(3*A[i]**2))**3 \
                            - 27*(R[j]*(9*B[k]*P[l]*A[i] - 18*A[i]**2*C[m]**2 - 2*R[j]**2)/(27*A[i]**3))**2
                        
                        if delta < 0:
                            delta = -1
                        if delta > 0 :
                            delta = 1
                        if delta == np.isnan:
                            delta = -1000
                            
                        D[i, j, k, l, m] = delta
    
    indexA = []
    indexR = []
    indexB = []
    indexP = []
    indexC = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                for l in range(size):
                    for m in range(size):
                        if D[i, j, k , l , m] > 0:
                            indexA.append(i)
                            indexR.append(j)
                            indexB.append(k)
                            indexP.append(l)
                            indexC.append(m)
               
    idx = np.random.randint(0, len(indexA)-1)
    
    return [A[indexA[idx]], R[indexR[idx]], B[indexB[idx]], P[indexP[idx]], C[indexC[idx]]]


size = 20
val = val_param_3eq(size)

                        
                        
                        
                        
                        
