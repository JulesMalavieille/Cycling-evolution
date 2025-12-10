"""
Created on Sat Nov  1 09:59:59 2025

@author: Jules Malavieille
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.optimize import newton

# Paramètres du modèle
beta = 3
P = 7
c = 1

# Paramètres fitness
a = 0.4
Rmax = 15
nu = 0.2
k = 1


R = lambda x : Rmax * np.exp(np.clip(-a*x, -50, 50))  # Force a etre max entre exp-50, exp50
Rp = lambda x : -a*R(x)
alpha = lambda deltaX : c*(1 - 1/(1 + nu * np.exp(-k*(deltaX))))
alphap0 = lambda : -(c*k*nu)/(1+nu)**2

# Rpp = lambda x : a**2*R(x)
# alphapp0 = lambda : (c*k**2*nu*(1-nu))/(1+nu)**3

# gN = lambda N : N / (c*c + N*N)
# gpN = lambda N : (c*c - N*N) / (c*c + N*N)**2
# gppN = lambda N : -2.0*N*(3.0*c*c - N*N) / (c*c + N*N)**3


def sol_eq(x):
    """
    Renvoie N*(x) : racine réelle ≥0 et STABLE de
      -α N^3 + r N^2 - (α c^2 + βP) N + r c^2 = 0
    pick: "stable_max" | "stable_min" 
    """
    r = float(R(x))
    alph = float(alpha(0.0))

    # coeffs du cubique (ordre décroissant)
    coeffs = np.array([-alph, r, -(alph*c**2 + beta*P), r*c**2], dtype=float)

    roots = np.roots(coeffs)
    # réelles ≥ 0 (tolérance)
    reals = np.array([z.real for z in roots if np.isfinite(z.real)
                      and abs(z.imag) < 1e-10 and z.real >= -1e-12], dtype=float)


    if len(reals) == 0:
        return float(0.0)
    else:
        return reals


# Simulation
x_init = 2
N = sol_eq(x_init)[2]
    

def canonique(t, Y):
    global N
    x = float(Y[0])
    
    u = 0.001
    sig = 0.01
    
    val = sol_eq(x)
    if len(val) == 3:
        
        if N > val[1]:
            N_bar = val[0]
            N = N_bar
        
        if N < val[1]:
            N_bar = val[2]
            N = N_bar
   
    else:
        N_bar = val[0]
        N = N_bar

    dr = Rp(x)
    dalpha = alphap0()
    
    dx = 1/2*u*sig * N_bar * (dr - dalpha*N_bar)
    
    return dx
    


def evol(x_init, time):
    global N
    tf = time[-1]
    S_ev = solve_ivp(canonique, [0,tf], [x_init], t_eval=time, method="LSODA")
    return S_ev.y[0]


####################################################
# Visualisation de la dynamique évolutive du trait x
####################################################
tf = 3000000
n = 30000000
time = np.linspace(0, tf, n)

E = evol(x_init, time)

plt.figure()
plt.plot(time, E)
plt.xlabel("Temps")
plt.ylabel("Valeur du trait x")
plt.title("Dynamique évolutive du trait x")
plt.grid()
