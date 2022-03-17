 # -*- coding : utf -8 -*-
"""
Created on Wed Mar 3 17:26:11 2021

@author : jmbru
"""

import pandas as pd
import matplotlib . pyplot as plt
import numpy as np
import scipy . stats
import scipy



def max_likelihood ( parameters ,data , sigma_u_2 ):
     omega = parameters [0]
     phi =np.exp ( parameters [1]) /(1+ np. exp ( parameters [1]) )
     sigma_eta =np. exp ( parameters [2])
     v = np. zeros ( len ( data ))
     a = np. zeros ( len ( data ))
     P = np. zeros ( len ( data ))
     F = np. zeros ( len ( data ))
     v,a,P,F,k= kalman_filter (data , sigma_u_2 , sigma_eta ,omega , phi )
     value =-( len ( data ))/2 * np. log (2* np.pi) - 0.5* np. sum(np. log (F) +(( v **2) /F))
     return -1* value

def max_likelihood2( parameters ,data , sigma_u_2 ,RV):
    omega = parameters [0]
    phi =np.exp ( parameters [1]) /(1+ np. exp ( parameters [1]) )
    sigma_eta =np. exp ( parameters [2])
    beta = parameters [3]
    v,a,P,F,k= kalman_filter_RV (data , sigma_u_2 , sigma_eta ,omega ,phi ,beta ,RV)
    value =-( len ( data ))/2 * np. log (2* np.pi) - 0.5* np. sum(np. log (F) +(( v **2) /F))
    return -1* value

def optimize_variances ( data ):
    result = scipy.optimize.minimize(max_likelihood,[-0.0457954, 0.99598032,  0.01142509], args =(data ,(np.pi ** 2) / 2) ,method ='BFGS')
    omega ,phi , sigma = result .x
    print ( result )
    print (omega ,np. exp( phi ) /(1+ np.exp( phi )),np. exp ( sigma ))
    return omega ,np. exp ( phi ) /(1+ np.exp( phi )),np. exp ( sigma )


def optimize_variances2 (data ,RV):
    result = scipy . optimize . minimize ( max_likelihood2 ,[1 , 1, 1 ,1], args =( data ,( np.pi **2) /2, RV), method ='BFGS')
    omega ,phi ,sigma , beta = result .x
    print ( result )
    print (omega ,np. exp( phi ) /(1+ np.exp( phi )),np. exp ( sigma ),beta )
    return omega ,np. exp ( phi ) /(1+ np.exp( phi )),np. exp ( sigma ),beta


def kalman_filter (data , sigma_u_2 , sigma_eta ,omega ,phi):
    v = np. zeros ( len ( data ))
    a = np. zeros ( len ( data ))
    P = np. zeros ( len ( data ))
    F = np. zeros ( len ( data ))
    a [0] = data [0]
    P [0] = 0
    v [0]= data [0] -a [0]
    F [0]= P [0]+ sigma_u_2
    for t in range (1, len( data )):
        att =a[t -1]+ P[t -1]* v[t -1]/ F[t -1]
        a[t]= phi *att + omega
        Ptt =P[t -1] -(P[t -1]**2) /F[t -1]
        P[t]= phi **2 * (Ptt) + sigma_eta **2
        F[t]=P[t]+ sigma_u_2
        if np. isnan ( data [t]):
            v[t ]=0
        else :
            v[t]= data [t]-a[t]
    return v,a,P,F,P/F


def smoothing_filter (data ,k,a,F,v,P):
    L = 1- k
    r = np. zeros ( len ( data ))
    alpha = np. zeros ( len( data ))
    V = np. zeros ( len ( data ))
    N = np. zeros (len( data ))
    for t in range ( len ( data ) -1,-1, -1):
        if np. isnan ( data [t]):
            r[t -1]= r[t]
            N[t -1]= N[t]
        else :
            r[t -1] = v[t]/F[t] + L[t]*r[t]
            N[t -1] = 1/F[t] +L[t ]**2 * N[t]
            alpha [t]=a[t]+P[t]*r[t -1]
            V[t] =P[t]- P[t ]**2 *N[t -1]
    return alpha ,V,r,N

def kalman_filter_RV (data , sigma_u_2 , sigma_eta ,omega ,phi ,beta ,RV):
    v = np. zeros(len(data))
    a = np. zeros(len(data))
    P = np. zeros(len(data))
    F = np. zeros(len(data))
    data = data
    a [0] = data [0] - beta *np.log (RV [0])
    P [0] = 0
    v [0]= data [0] -a [0]
    F [0]= P [0]+ sigma_u_2
    for t in range (1, len( data )):
        att =a[t -1]+ P[t -1]* v[t -1]/ F[t -1]
        a[t]= phi *att + omega - beta *np. log(RV[t])
        Ptt =P[t -1] -(P[t -1]**2) /F[t -1]
        P[t]= phi **2 * (Ptt) + sigma_eta **2
        F[t]=P[t]+ sigma_u_2
        if np. isnan ( data [t]):
            v[t ]=0
        else :
            v[t]= data [t]-a[t]
    return v,a,P,F,P/F


def smoothing_filter_RV (data ,k,a,F,v,P,beta ,RV):
    data =data - beta *np.log (RV)
    L = 1- k
    r = np. zeros ( len ( data ))
    alpha = np. zeros ( len( data ))
    V = np. zeros ( len ( data ))
    N= np. zeros (len( data ))
    for t in range ( len ( data ) -1,-1, -1):
        if np. isnan ( data [t]):
            r[t -1]= r[t]
            N[t -1]= N[t]
        else :
            r[t -1] = v[t]/F[t] + L[t]*r[t]
            N[t -1] = 1/F[t] +L[t ]**2 * N[t]
            alpha [t]=a[t]+P[t]*r[t -1]
            V[t] =P[t]- P[t ]**2 *N[t -1]
    return alpha ,V,r,N


def calculate_returns ( data ):
    returns =[]
    for i in range (1, len( data )):
        returns . append (np.log ( data [i]/ data [i -1]) )
    return returns


def ImportData ():
    data = pd.read_excel('SvData.xlsx')
    #data = data.rename(columns = {'// Pound / Dollar daily exchange rates \ sections 9.6 and 14.4 ': 'Level' })
    #array = np.array(data)
    #data = array[: ,0]
    #print(data)
    return(data["GBPUSD"].values)




def TransformData ( data ):
    return np. log (( data -np. mean ( data )) **2) -1.27

def main ():
    sigma_u_2 = (np.pi **2) /2
    data = ImportData().astype(np.float)
    data = data /100 - np.mean(data /100)
    
    plt.figure()
    plt.plot(data)
    plt.xlabel(" Observation ")
    plt.ylabel(" Pound dollar exchange rate ")
    plt.savefig("data")
    plt.show()

    plt.figure()
    transformed_data =np. log ( data **2)
    plt.plot( transformed_data ,'o')
    plt.xlabel(" Observation ")
    plt.ylabel("Log (y_t ^2) ")
    plt.savefig(" Transformed data ")
    plt.show ()
    
    omega ,phi , sigma_eta = optimize_variances (( transformed_data -1.27) )
    #omega ,phi , sigma_eta = -0.0457954, 0.99598032,  0.01142509
    shift_transformed_data = transformed_data -1.27
    
    v,a,P,F,K= kalman_filter ( shift_transformed_data , sigma_u_2 , sigma_eta ,omega ,phi)
    alpha ,V,r,N= smoothing_filter ( shift_transformed_data ,K,a,F,v,P)
    
    plt.figure()
    plt.plot( transformed_data ,'o',label =" Log (y_t ^2) ")
    plt.plot(alpha , label =" Smoothed estimate ")
    plt.legend ()
    plt.xlabel(" Observation ")
    plt.savefig(" Transformed data ")
    plt.show()
    
    plt.figure()
    plt.plot(a- omega /(1 - phi),label =" Filtered estimate based H_t ")
    plt.plot(alpha - omega /(1 - phi ),label =" Smoothed estimate based H_t")
    plt.legend()
    plt.xlabel(" Observation ")
    plt.savefig(" H_t estimates ")
    plt.show()


    SPX =np.genfromtxt (" SPX . csv ", delimiter =',', dtype = float , skip_header = True )
    prices = SPX [: ,17]
    RV=SPX[: ,4][1:]
    prices =np. array ( calculate_returns ( prices ))
    shift_transformed_prices = TransformData ( prices )
    
    omega_SPX , phi_SPX , sigma_eta_SPX = optimize_variances((shift_transformed_prices))
    omega_SPX_RV, phi_SPX_RV, sigma_eta_SPX_RV, beta_SPX_RV = optimize_variances2 (shift_transformed_prices ,RV)
    v_SPX, a_SPX, P_SPX , F_SPX, K_SPX = kalman_filter(shift_transformed_prices, sigma_u_2, sigma_eta_SPX, omega_SPX, phi_SPX)

    alpha_SPX ,V_SPX ,r_SPX , N_SPX = smoothing_filter ( shift_transformed_prices ,K_SPX ,a_SPX ,F_SPX ,
    v_SPX , P_SPX )
    
    v_SPX_RV , a_SPX_RV , P_SPX_RV , F_SPX_RV , K_SPX_RV = kalman_filter_RV ( shift_transformed_prices ,sigma_u_2 , sigma_eta_SPX_RV , omega_SPX_RV , phi_SPX_RV , beta_SPX_RV ,RV)
    alpha_SPX_RV , V_SPX_RV , r_SPX_RV , N_SPX_RV = smoothing_filter_RV ( shift_transformed_prices , K_SPX_RV , a_SPX_RV , F_SPX_RV , v_SPX_RV , P_SPX_RV , beta_SPX_RV ,RV)
    
    plt.plot(shift_transformed_prices ,'o',label ="Log( y_t ^2) ")
    plt.plot(alpha_SPX , label =" Smoothed estimate SPX ")
    plt.plot(alpha_SPX_RV , label =" Smoothed estimate SPX with RV")
    plt.legend ()
    plt.xlabel (" Observation ")
    plt.savefig (" Transformed data SPX ")
    plt.show ()
    
    
    plt.plot (a_SPX - omega_SPX /(1 - phi_SPX ),label =" Filtered estimate based H_t ")
    plt.plot ( alpha_SPX - omega_SPX /(1 - phi_SPX ),label =" Smoothed estimate based H_t ")
    plt.xlabel (" Observation ")
    plt.savefig (" H_t estimates SPX ")
    plt.legend ()
    plt.show ()
    
    plt.plot(a_SPX_RV -( omega_SPX_RV )/(1 - phi_SPX_RV ),label =" Filtered estimate based H_t")
    plt.plot(alpha_SPX_RV -( omega_SPX_RV )/(1 - phi_SPX_RV ),label =" Smoothed estimate based H_t ")
    plt.xlabel(" Observation ")
    plt.savefig(" H_t estimates SPX with RV")
    plt.legend()
    plt.show()

main()
