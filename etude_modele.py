"""
Created on Wed Oct  29 13:30:29 2025

@author: Jules Malavieille
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.optimize import newton


# Paramètres du modèle
beta = 3
P = 7
c = 1

# Paramètres fitness
a = 0.4
Rmax = 15
nu = 0.5
k = 1

pick = "N4"

R = lambda x : Rmax * np.exp(np.clip(-a*x, -50, 50))  # Force a etre max entre exp-50, exp50
alpha = lambda deltaX : c*(1 - 1/(1 + nu * np.exp(-k*(deltaX))))

Rp = lambda x : -a*R(x)
alphap0 = lambda : -(c*k*nu)/(1+nu)**2


def eq(x):
    """
    Renvoie N*(x) : racine réelle ≥0 et STABLE de
      -α N^3 + r N^2 - (α c^2 + βP) N + r c^2 = 0
    pick: "N4" | "N2" 
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
    if pick == "N2":
        return float(min(reals))
    # défaut: plus grand équilibre stable (souvent le « haut »)
    return float(max(reals))


def eq2(x):
    N_min = 0
    N_max = 100
    nb_val = 1000

    N = np.linspace(N_min, N_max, nb_val)
    Y = []
    N_sol = []
    
    r = R(x)
    alph = alpha(0)
    for i in range(len(N)):
        y = -alph*N[i]**3 + r*N[i]**2 + N[i]*(-alph*c**2 - beta*P) + r*c**2
        Y.append(y)
        
        if Y[i-1] < 0 and Y[i] > 0:
            N_sol.append((N[i-1]+ N[i])/2)
        if Y[i-1] > 0 and Y[i] < 0:
            N_sol.append((N[i-1]+ N[i])/2)
        if Y[i] == 0:
            N_sol.append(N[i])
                   
    return N_sol

        
def grad_fit(x):
    rp = Rp(x)
    alphp0 = alphap0()
    N_eq = eq(x)
    df = rp - alphp0*N_eq
    return df 


def delta(x):
    delt = -4*(c**2 + beta*P/alpha(0) - R(x)**2/(3*alpha(0)**2))**3 \
        - 27*(R(x)*(9*beta*P*alpha(0) - 18*alpha(0)**2*c**2 - 2*R(x)**2)/(27*alpha(0)**3))**2
    return delt
        

def crit():
    critN2 = None
    critN4 = None
    
    try :
        critN4 = newton(delta, 0)
    except RuntimeError:
        pass
    
    try :
        critN2 = newton(delta, 3)
    except RuntimeError:
        pass
    
    if critN2 is not None and critN4 is not None:
        return critN2, critN4
    
    if critN2 is not None or critN4 is not None:
        return 0, 5


#############################################################################
# Récupération des valeurs d'équilibre démographique en fonction de x et r(x)
#############################################################################
XL = np.linspace(0, 7, 1000)
Neq = []
N1eq = []
N2eq = []
X1 = []
X2 = []
for i in range(len(XL)):
    N_sol = eq2(XL[i])
    if len(N_sol) == 3:
        Neq.append(N_sol[1])
        N1eq.append(N_sol[0])
        X1.append(XL[i])
        N2eq.append(N_sol[2])
        X2.append(XL[i])
    else:
        Neq.append(eq(XL[i]))

critN2, critN4 = crit()
    
####################################################################
# Valeur Différente section du modèle en fonction des valeurs de r(x) 
####################################################################
# plt.figure()
# plt.plot(Rx, Neq)
# plt.plot(Rx1, N1eq, "--")
# plt.plot(Rx2, N2eq, "--")
# plt.axvline(2.095, color="k")
# plt.axvline(3.729, color="k")
# plt.axvline(10.67, color="k")
# plt.xlabel("Valeur de r")
# plt.ylabel("Valeur d'équilibre de N")
# plt.grid()


##################################################################
# Valeur Différente section du modèle en fonction des valeurs de x 
##################################################################
plt.figure()
plt.plot(XL, Neq)
plt.plot(X1, N1eq, "--")
plt.plot(X2, N2eq, "--")
plt.axvline(critN4, color="k")
plt.axvline(critN2, color="k")
plt.xlabel("Valeur de X")
plt.ylabel("Valeur d'équilibre de N")
plt.title("Dynamique du modèle en fonction de x")
plt.grid()


#######################################################################################
# Valeur de gradient de fitness évolutive pour les différents équilibres démographiques
#######################################################################################
XL = np.linspace(0, 5, 1000)
RX = R(XL)
S1 = []
X1 = []
S2 = []
X2 = []
R1 = []
R2 = []
for i in range(len(XL)):
    if critN2 != 0:
        if critN4 < XL[i] <= 5:
            s1 = grad_fit(XL[i])
            S1.append(s1)
            X1.append(XL[i])
            R1.append(RX[i])
        
        if 0 <= XL[i] < critN2 :
            s2 = grad_fit(XL[i])
            S2.append(s2)
            X2.append(XL[i])
            R2.append(RX[i])

    

plt.figure()
plt.plot(X1, S1, color="blue", alpha=0.7, label="Gradient de fitness évolutive domaine de N2")
plt.plot(X2, S2, color="orange", alpha=0.7, label="Gradient de fitness évolutive domaine de N4")
plt.axhline(0, color="k")
plt.xlabel("Valeur de x")
plt.ylabel("Valeur de gradient de la fitness évolutive")
plt.title("Valeur de gradient de fitness (si équilibres stables multiples : " + str(pick) +")")
plt.grid()
plt.legend()

