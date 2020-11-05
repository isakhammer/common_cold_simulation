import numpy as np

print("hello")


# Grid
th_0 = 0.25
th_1 = 0.50
n_a = 6
th_a = np.linspace(th_0, th_1,  n_a  )
th_a = th_a[:, np.newaxis]
I_a  = np.ones((n_a,1))
H_a  = np.abs(th_a@I_a.T - I_a@ th_a.T)
E_y = 0.5
sigma_y = 0.5**2


# Given data
y_b =     np.array([ 0.5, 0.32, 0.40, 0.35, 0.60])
th_b =  np.array([ 0.3, 0.35, 0.39, 0.41, 0.45])
n_b = th_b.shape[0]
th_b = th_b[:, np.newaxis]
I_b = np.ones(( n_b, 1))
H_b = np.abs(th_b@I_b.T - I_b@ th_b.T)
E_y_b = np.mean(y_b)

# Internal difference
H_ab = np.abs(th_a @I_b.T  - I_a @ th_b.T)

def corr(d):
    return (1 + 15*np.abs(d))*np.exp(-15*d)

# Build covariance matrix
sigma_a = 1
sigma_b = 1
Sigma_a = corr(H_a)
Sigma_b = corr(H_b)
Sigma_ab = corr(H_ab)
