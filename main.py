
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew
import math as ms
from scipy.stats import norm

def get_xt_yt():
    df = pd.read_excel("SvData.xlsx")
    svdata = df["GBPUSD"].values
    y_t = svdata / 100
    y_t_demeaned = y_t - np.mean(y_t)
    x_t = np.log(y_t_demeaned ** 2)
    return  y_t, y_t_demeaned, x_t


def estimate_params(x_t):
    a, b, c = np.mean(x_t), np.var(x_t), np.cov(x_t[1:],x_t[:-1])[0][1]
    estimate_phi        = c /(b - (np.pi **2) /2)
    if estimate_phi >=1:
        estimate_phi = 0.999
    estimate_sigma_eta = np.sqrt((c * (1 - estimate_phi **2))/ estimate_phi)
    estimate_omega = (1 - estimate_phi) * (a + 1.27)
    return estimate_phi, estimate_sigma_eta, estimate_omega


def inverse(M):
    """
    Take inverse of a matrix M
    """
    if M.size == 1:
        res = np.array([[1 / M[0][0]]])
    else:
        res = np.linalg.inv(M)
        
    return(res)

def Kalman_Filter(y, a1, p1, sigma_eta_hat, phi_hat, omega_hat):
    """
    Kalman filter for the general state space model
    see slide 21 of lecture slides of week 3 
    """
    n = len(y)
    
    # create empty arrays 
    a = np.zeros((n + 1,1,1))
    a[0] = np.array([[a1]])
    P = np.zeros((n + 1,1,1))
    P[0] = np.array([[p1]])
    (v_t, F_t, K_t, L_t, q005, q095) = [np.zeros((n,1,1)) for i in range(6)]
    
    
    # System matrices of the general state space form 
    Z_t = np.array([[1]])
    H_t = np.array([[(np.pi ** 2) /2]])
    T_t = np.array([[phi_hat]])
    R_t = np.array([[1]])
    Q_t = np.array([[sigma_eta_hat ** 2]])
    
    # c_t (intercept in the observation equation) and d_t (intercept in the state equation) 
    c_t = -1.27
    d_t = omega_hat
    
    for i in range(n):         
        v_t[i]  = y[i] - c_t - np.dot(Z_t, a[i])
        F_t[i]  = np.dot(np.dot(Z_t, P[i]), Z_t.T) + H_t
        K_t[i]  = np.dot(np.dot(np.dot(T_t, P[i]), Z_t.T), inverse(F_t[i]))
        L_t[i]  = T_t - np.dot(K_t[i], Z_t)
        q005[i] = np.array([[norm.ppf(0.05, loc = float(a[i]), scale = ms.sqrt(float(P[i])))]])
        q095[i] = np.array([[norm.ppf(0.95, loc = float(a[i]), scale = ms.sqrt(float(P[i])))]])
        a[i + 1]   = d_t + np.dot(T_t, a[i]) + np.dot(K_t[i], v_t[i])
        P[i + 1]   = np.dot(np.dot(T_t, P[i]), T_t.T) + np.dot(np.dot(R_t, Q_t), R_t.T) - np.dot(np.dot(K_t[i], F_t[i]), K_t[i].T)

    return a, P, v_t, F_t, K_t, L_t, q005, q095, n

def  Kalman_Smoother(n, v_t, F_t, L_t, a, P, y, K_t, phi_hat, sigma_eta_hat, omega_hat):
      """
      Input are the sequences produced during the Kalman Filter process,
      but returns the backward recursion Kalman Smoothing results 
      """
      Z_t = np.array([[1]]) 
      H_t = np.array([[(np.pi ** 2) / 2]])
      Q_t = np.array([[sigma_eta_hat ** 2]])
     
      r_t        = np.zeros((n,1,1))
      r_t[n - 1] = np.array([[0]])
      alpha_hat  = np.zeros((n,1,1))
     
      N_t        = np.zeros((n,1,1))
      N_t[n - 1] = np.array([[0]])
      V_t        = np.zeros((n,1,1))
      V_t[n - 1] = np.array([[0]])
      q005 = np.zeros((n,1,1))
      q095 = np.zeros((n,1,1))
     
      eps_hat    = np.zeros((n,1,1))
      var_eps_yn = np.zeros((n,1,1))
      eta_hat    = np.zeros((n,1,1))
      var_eta_yn = np.zeros((n,1,1))
      D          = np.zeros((n,1,1))
      
      for j in range(n-1,-1,-1):
         r_t[j - 1]   = np.dot(v_t[j], inverse(F_t[j])) + np.dot(L_t[j], r_t[j])
         alpha_hat[j] = a[j] + np.dot(P[j], r_t[j - 1])
         N_t[j - 1]   = np.dot(np.dot(Z_t.T, inverse(F_t[j])), Z_t) + np.dot(np.dot(L_t[j].T, N_t[j]), L_t[j])
         V_t[j]       = P[j] - np.dot(np.dot(P[j], N_t[j - 1]), P[j]) 
         q005[j] = np.array([[norm.ppf(0.05, loc = float(alpha_hat[j]), scale = ms.sqrt(float(V_t[j])))]])
         q095[j] = np.array([[norm.ppf(0.95, loc = float(alpha_hat[j]), scale = ms.sqrt(float(V_t[j])))]])
             
         eps_hat[j]   = y[j] - alpha_hat[j]
         D[j]         = inverse(F_t[j]) + np.dot(np.dot(K_t[j], K_t[j]), N_t[j]) # equation 2.47
         var_eps_yn[j]= H_t - np.dot(np.dot(H_t, H_t), D[j])
         eta_hat[j]   = np.dot(Q_t, r_t[j])
         var_eta_yn[j]= Q_t - np.dot(np.dot(Q_t, Q_t), N_t[j]) # equation 2.47
         
      return r_t, alpha_hat, N_t, V_t, q005, q095, eps_hat, var_eps_yn, eta_hat, var_eta_yn, D
  


def Plot_a_b(y_t, x_t):
    
    ### Plot for point (a)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 3)
    
    ax1.hlines(y=0, xmin = -1, xmax = len(y_t)+1, linestyle='-', color = 'black', lw =1)
    # plt.set_ylim(-0.0375, 0.05)
    ax1.set_ylim(-0.0375, 0.05)
    ax1.set_xlim(-5, 947)
    #ax1.set_ylabel('Demeaned Returns')
    ax1.set_xlabel('Time')
    ax1.plot(y_t, color = 'slateblue', lw  =1)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(1.5)
        else:
            ax1.spines[axis].set_visible(False)
    
    fig, ax2 = plt.subplots()
    fig.set_size_inches(10, 3)
    ax2.plot(x_t, color = 'slateblue', lw  =1)
    ax2.set_ylim(-30, -6)
    ax2.set_xlim(-5, 947)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(1.5)
        else:
            ax2.spines[axis].set_visible(False)
  


def Plot_KF(x_t, a, ftsize, lw): 
    """
    Plots the Kalman Filter 
    """
    
    fig1, (ax1, ax2) = plt.subplots(2,1)
    fig1.set_size_inches(12, 6)
    a = [float(el) for el in a]
    
    t = np.array([i for i in range(1,945+1)])
    
    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.scatter(t, x_t, color = "darkslateblue", s = 12)
    ax1.plot(t[1:], a[1:-1])
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax1.set_xlim(-5, 947)
    ax1.set_ylim(-29.477529750730948, -5.061570924204879)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)
            
    # SUBPLOT 1 upper left ------------------------------------------------
    #ax1.scatter(t, x_t, color = "darkslateblue", s = 12)
    ax2.plot(t[1:], a[1:-1])
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax2.set_xlim(-5, 947)
    ax2.set_ylim(-11, -8)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)



def Plot_KS(x_t, alpha_hat, ftsize, lw): 
    """
    Plots the Kalman Smoother
    """
    
    fig1, (ax1, ax2) = plt.subplots(2, 1)
    fig1.set_size_inches(12, 6)
    alpha_hat = [float(el) for el in alpha_hat]
    
    t = np.array([i for i in range(1,945+1)])
    ax1.scatter(t, x_t, color = "darkslateblue", s = 12)
    ax1.plot(alpha_hat)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax1.set_xlim(-5, 947)

    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)
            
    ax2.plot(alpha_hat)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax2.set_xlim([-5, 947])
    ax2.set_ylim(-11, -8)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
            
            
def Plot_H(H_filter, H_smoother, ftsize, lw): 
    """
    Plots the Kalman Smoother
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(10, 3)
    H_filter   = [float(el) for el in H_filter]
    H_smoother = [float(el) for el in H_smoother]
    
    t = np.array([i for i in range(1,945+1)])
    ax1.plot(t, H_filter, lw = 1 )
    ax1.plot(t, H_smoother, lw  = 1)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)

    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)


def main():
    y_t, y_t_demeaned, x_t = get_xt_yt()
    
    phi_hat, sigma_eta_hat, omega_hat = estimate_params(x_t) 
    a1 = 0
    p1 = 10 ** 7
    a, P, v_t, F_t, K_t, L_t, q005, q095, n = Kalman_Filter(x_t, a1, p1, sigma_eta_hat, phi_hat, omega_hat)
    r_t, alpha_hat, N_t, V_t, q005, q095, eps_hat, var_eps_yn, eta_hat, var_eta_yn, D = Kalman_Smoother(n, v_t, F_t, L_t, a, P, x_t, K_t, sigma_eta_hat, phi_hat, omega_hat)
    print(len(alpha_hat))

    H_filter = np.array([float(el) for el in  a[:-1]]) - x_t
    H_smoother = np.array([float(el) for el in  alpha_hat]) - x_t
    
    
    print("estimate phi         = ", np.round(phi_hat, 3))
    print("estimate omega       = ", np.round(omega_hat, 3))
    print("estimate sigma_eta   = ", np.round(sigma_eta_hat, 3))
    
    #Plot_a_b(y_t_demeaned, x_t)
    Plot_KF(x_t = x_t, a = a, ftsize = 12, lw =1.5)
    Plot_KS(x_t = x_t, alpha_hat = alpha_hat, ftsize = 12, lw =1.5)
    Plot_H(H_filter,H_smoother, ftsize =12, lw =1.5)

    
    
if __name__ == "__main__":
    main()
