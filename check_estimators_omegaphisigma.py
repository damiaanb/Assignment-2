import numpy as np
from scipy.stats import chi2



def simulatedata(n, omega, phi, sigma_eta):
    
    x = np.zeros(n)
    h = np.zeros(n+1)
    for i in range(n):
        h[i+1] = omega + phi * h[i] + sigma_eta * float(np.random.normal(0, 1, 1))
        x[i]   = h[i] + float(np.log(chi2.rvs(1, size=1)))
        
    return x, h[1:]


omega, phi, sigma_eta = 0.2, 0.999, 1.77
x, h = simulatedata(100000, omega, phi, sigma_eta) # simulate from equation (2) of the assignment 

a = np.mean(x)                 # sample mean
b = np.var(x)                  # sample variance
c = np.cov(x[1:],x[:-1])[0][1] # sample covariance


estimate_phi        = c /(b - (np.pi **2) /2)
estimate_sigma_eta  = np.sqrt((c * (1 - estimate_phi **2))/ estimate_phi)
estimate_sigma_eta_2  = np.sqrt((b  - (np.pi **2) /2) * (1 - estimate_phi **2))
estimate_omega = (1 - estimate_phi) * (a + 1.27)


print("real     phi           = ", phi)
print("estimate phi           = ", np.round(estimate_phi, 2))
print("real     omega         = ", omega)
print("estimate omega         = ", np.round(estimate_omega, 2))
print("real     sigma_eta     = ", sigma_eta)
print("estimate sigma_eta     = ", np.round(estimate_sigma_eta, 2))
print("estimate sigma_eta_2   = ", np.round(estimate_sigma_eta_2, 2))