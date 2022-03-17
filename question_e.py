 

import pandas as pd 
import numpy as np
from main import mm_estimate, Find_MLE
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def mm_estimate(x_t):
    """
    Get the MM-estimates of phi, sigma_eta, and omega for initial values of QML
    """
    a, b, c = np.mean(x_t), np.var(x_t), np.cov(x_t[1:],x_t[:-1])[0][1]
    phi_0        = c /(b - (np.pi **2) /2)
    if phi_0 >=1:
        phi_0 = 0.999
    sigma_eta_0 = np.sqrt((c * (1 - phi_0 **2))/ phi_0)
    omega_0 = (1 - phi_0) * (a + 1.27)
    return (phi_0, sigma_eta_0, omega_0)



def Log_L(params, data):
    a, P, v, F, K, L, n = Kalman_Filter(data, params[0], params[1], params[2])
    v = np.array([float(el) for el in v]) 
    F = np.array([float(el) for el in F]) 
    value =-(n/2) * np.log(2 * np.pi) - 0.5 * np.sum(np.log(F) + ((v ** 2) / F))
    return -1 * value
 
def Find_MLE(data, init):
    result = minimize(Log_L, init, args =(data), bounds =((-1,1),(0, 10),(None, None)))
    print(result)
    phi, sigma, omega = result.x
    return phi, sigma, omega

def inverse(M):
    """
    Take inverse of a matrix M
    """
    if M.size == 1:
        res = np.array([[1 / M[0][0]]])
    else:
        res = np.linalg.inv(M)
        
    return(res)


def Kalman_Filter(y, phi, sigma_eta, omega):
    """
    Kalman filter for the general state space model
    see slide 21 of lecture slides of week 3 
    """
    n = len(y)
    
    # create empty arrays 
    a = np.zeros((n + 1,1,1))
    a[0] = np.array([[y[0]]])
    P = np.zeros((n + 1,1,1))
    P[0] = np.array([[0]])
    (v_t, F_t, K_t, L_t, q005, q095) = [np.zeros((n,1,1)) for i in range(6)]
    
    
    # System matrices of the general state space form 
    Z_t = np.array([[1]])
    H_t = np.array([[(np.pi ** 2) /2]])
    T_t = np.array([[phi]])
    R_t = np.array([[1]])
    Q_t = np.array([[sigma_eta ** 2]])
 
    for i in range(n):
        if (np.isnan(y[i])):
            
            v_t[i]     = np.array([[0]])
            F_t[i]  = np.dot(np.dot(Z_t, P[i]), Z_t.T) + H_t
            K_t[i]  = np.dot(np.dot(np.dot(T_t, P[i]), Z_t.T), inverse(F_t[i]))
            L_t[i]  = T_t - np.dot(K_t[i], Z_t)
            a[i + 1]   = np.dot(T_t, a[i]) 
            P[i + 1]   = np.dot(np.dot(T_t, P[i]), T_t.T) + np.dot(np.dot(R_t, Q_t), R_t.T) 
            
        else:
        
            v_t[i]  = y[i] - np.dot(Z_t, a[i])
            F_t[i]  = np.dot(np.dot(Z_t, P[i]), Z_t.T) + H_t
            K_t[i]  = np.dot(np.dot(np.dot(T_t, P[i]), Z_t.T), inverse(F_t[i]))
            L_t[i]  = T_t - np.dot(K_t[i], Z_t)
            a[i + 1] = np.dot(T_t, a[i]) + np.dot(K_t[i], v_t[i]) + omega
            P[i + 1] = np.dot(np.dot(T_t, P[i]), T_t.T) + np.dot(np.dot(R_t, Q_t), R_t.T) - np.dot(np.dot(K_t[i], F_t[i]), K_t[i].T)

    return a, P, v_t, F_t, K_t, L_t, n

def Kalman_Smoother(n, v_t, F_t, L_t, a, P, y, K_t):
      """
      Input are the sequences produced during the Kalman Filter process,
      but returns the backward recursion Kalman Smoothing results 
      """
      Z_t = np.array([[1]]) 
      
      r_t        = np.zeros((n,1,1))
      r_t[n - 1] = np.array([[0]])
      alpha_hat  = np.zeros((n,1,1))
     
      N_t        = np.zeros((n,1,1))
      N_t[n - 1] = np.array([[0]])
      V_t        = np.zeros((n,1,1))
      V_t[n - 1] = np.array([[0]])

      
      for j in range(n-1,-1,-1):
         r_t[j - 1]   = np.dot(v_t[j], inverse(F_t[j])) + np.dot(L_t[j], r_t[j])
         alpha_hat[j] = a[j] + np.dot(P[j], r_t[j - 1])
         N_t[j - 1]   = np.dot(np.dot(Z_t.T, inverse(F_t[j])), Z_t) + np.dot(np.dot(L_t[j].T, N_t[j]), L_t[j])
         V_t[j]       = P[j] - np.dot(np.dot(P[j], N_t[j - 1]), P[j]) 
         
      return r_t, alpha_hat, N_t, V_t
  
    

def get_sp500():
    df =  pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
    df_SPX =  df.loc[df['Symbol'] == '.SPX'].reset_index(drop = True)
    df_ajax = df_SPX.loc[(len(df_SPX)- 1248):len(df_SPX)]
    close_price = df_ajax["close_price"].values
    RV = df_ajax["rk_twoscale"]
    returns = np.log(close_price[1:] / close_price[:-1]) 
    return returns, RV 


def Plot_KF(x_t, a, ftsize, lw): 
    """
    Plots the Kalman Filter 
    """
    
    fig1, (ax1, ax2) = plt.subplots(2,1)
    fig1.set_size_inches(12, 6)
    a = [float(el) for el in a]
    
    t = np.array([i for i in range(1,945+1)])
    
    # SUBPLOT 1 above -----------------------------------------------------
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
            
    # SUBPLOT 2 below  ---------------------------------------------------
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
    
    fig1, ax1 = plt.subplots(1, 1)
    fig1.set_size_inches(10, 3)
    alpha_hat = [float(el) for el in alpha_hat]
    t = np.array([i for i in range(1,len(x_t)+1)])
    
    # SUBPLOT 1 above -----------------------------------------------------
    ax1.scatter(t, x_t, color = "darkslateblue", s = 30, alpha = 0.6, edgecolors = 'None')
    ax1.plot(alpha_hat, lw = lw)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    #ax1.set_xlim(-5, 947)

    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)


# y_t, y_t_demeaned, x_t = get_xt_yt() 
# Plot_a_b(y_t_demeaned, x_t) 

def Plot_H(H_filter, H_smoother, ftsize, lw): 
    """
    Plots the filtered (E(H_t | Y_t)) and smoothed (E(H_t | Y_n))
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(10, 3)
    H_filter   = [float(el) for el in H_filter]
    H_smoother = [float(el) for el in H_smoother]
    
    t = np.array([i for i in range(1,len(x_t)+1)])
    ax1.plot(t, H_filter, lw = lw, color = "darkslateblue" )
    ax1.plot(t, H_smoother, lw  = lw)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)

    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)

def main():
    y_t = get_sp500()
    y_t_demeaned = y_t - np.mean(y_t)
    x_t = np.log(y_t_demeaned ** 2)
    print(y_t)
    
if __name__ == "__main__":
    y_t, RV = get_sp500()
    y_t_demeaned = y_t - np.mean(y_t)
    x_t = np.log(y_t_demeaned ** 2)
    
    # SV MODEL 
    #omega_2 , phi_2 , sigma_eta_2 = optimize_variances(x)
    init = mm_estimate(x_t - 1.27)
    phi2, sigma_eta2, omega2 = Find_MLE(x_t - 1.27, init)
    print("  ")
    print("SV model")
    print("  ")
    print("Estimates are")
    print("phi         = ", phi2)
    print("sigma_eta   = ", sigma_eta2)
    print("omega       = ", omega2)
    print("  ")
    print("Rounded:")
    print("(phi, sigma_eta, omega) = ", "("+str(np.round(phi2,4))+",", str(np.round(sigma_eta2,4))+",", str(np.round(omega2,4))+")")
    a, P, v_t, F_t, K_t, L_t, n = Kalman_Filter(x_t - 1.27, phi2, sigma_eta2, omega2)
    r_t, alpha_hat, N_t, V_t = Kalman_Smoother(n, v_t, F_t, L_t, a, P, x_t, K_t)
    
    Plot_KS(x_t = x_t, alpha_hat = alpha_hat, ftsize = 12, lw =1.5)
    #Plot_H(H_filter,H_smoother, ftsize =12, lw =1.5)
    H_filter = np.array([float(el) for el in  a[:-1]]) - omega2 /(1 - phi2)
    H_smoother = np.array([float(el) for el in  alpha_hat]) - omega2 /(1 - phi2)
    Plot_H(H_filter,H_smoother, ftsize =12, lw =1.5)
    
    
    
    # SV MODEL with RV 
    #omega3, phi3, sigma_eta3, beta3 = optimize_variances2(x_t - 1.27, RV)