import numpy as np
import matplotlib.pyplot as plt



def gauss_process(t_a, mu_a, t_b, mu_b, x_b, sigma=0.5**2, phi=15):

    # Difference Matrix
    I_a  = np.ones(t_a.shape)
    H_a  = np.abs(mu_a@I_a.T - I_a@ mu_a.T)

    I_b = np.ones(x_b.shape)
    H_b = np.abs(mu_b@I_b.T - I_b@ mu_b.T)

    H_ab = np.abs(th_a @I_b.T  - I_a @ th_b.T)

    def corr(d):
        return (1 + phi*np.abs(d))*np.exp(-phi*d)

    # Build covariance matrix
    Sigma_a = corr(H_a)
    Sigma_b = corr(H_b)
    Sigma_ab = corr(H_ab)

    Sigma_b_inv = np.linalg.inv(Sigma_b)
    E_a_b = mu_a + Sigma_ab @ Sigma_b_inv @ ( x_b - E_y_b)
    var_a_b = Sigma_a - Sigma_ab @ Sigma_b_inv @ Sigma_ab.T

    return E_a_b, var_a_b

# Inital expected value
E = 0.5

# Grid
th_0 = 0.25
th_1 = 0.50
n = 51
th = np.linspace(th_0, th_1,  n  )
th = th[:, np.newaxis]
E_y = E*np.ones(th.shape)

# Given data
y_b =     np.array([ 0.5, 0.32, 0.40, 0.35, 0.60])
th_b =  np.array([   0.3, 0.35, 0.39, 0.41, 0.45])
th_b = th_b[:, np.newaxis]

E_y_b = np.mean(y_b)

E, Var = gauss_process(t_a=th, mu_a=E_y, t_b=th_b, mu_b=E_y_b, x_b=y_b)

plt.scatter( th_b, y_b, label = "Data")
plt.plot( th, E, label = "Expected values" )
plt.legend()
plt.show()
