import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def gauss_process(t_a, mu_a, mu_b, t_b, x_b, sigma=0.5**2, phi=15):

    # Difference Matrix
    I_a = np.ones(t_a.shape)
    I_b = np.ones(x_b.shape)

    H_a = np.abs(t_a @ I_a.T - I_a @ t_a.T)
    H_b = np.abs(t_b @ I_b.T - I_b @ t_b.T)
    H_ab = np.abs(t_a @ I_b.T - I_a @ t_b.T)

    def corr(d):
        return (1 + phi*d)*np.exp(-phi*d)

    # Build covariance matrix
    Sigma_a = corr(H_a)
    Sigma_b = corr(H_b)
    Sigma_ab = corr(H_ab)

    Sigma_b_inv = np.linalg.inv(Sigma_b)

    E_ab = mu_a + Sigma_ab @ Sigma_b_inv @ (x_b - mu_b)
    var_ab = Sigma_a - Sigma_ab @ Sigma_b_inv @ Sigma_ab.T
    return E_ab, var_ab


# Grid
t_0 = 0.25
t_1 = 0.50
E = 0.5
n = 51
th = np.linspace(t_0, t_1,  n  )
th = th[:, np.newaxis]
E_y = 0.5*np.ones(th.shape)

# Given data
y_b =     np.array([ 0.5, 0.32, 0.4, 0.40, 0.35, 0.60])
y_b = y_b[:, np.newaxis]
th_b =  np.array([   0.3, 0.35, 0.33, 0.39, 0.41, 0.450])
th_b = th_b[:, np.newaxis]
E_y_b = E*np.ones(y_b.shape)
E, Var = gauss_process(t_a=th, mu_a=E_y, t_b=th_b, mu_b=E_y_b, x_b=y_b)
E = E[:,0]
th = th[:,0]
Var_diag = np.diagonal(Var)


E_l = E - Var_diag*1.645
E_u = E + Var_diag*1.645
plt.scatter( th_b, y_b, label = "Data")
plt.plot( th, E, label = "Expected values" )
plt.fill_between(th, E_l, E_u, color="gray", alpha  = 0.5)
plt.xlabel(r"$\theta$", size=20)
plt.ylabel(r'$y(\theta)$', size=20)
plt.title("Mean vector and 90% Prediction interval", size=15)
plt.savefig("gp_2c.png")
plt.grid()
plt.legend()
plt.show()

Var_diag = np.diagonal(Var)
N = len(Var_diag)
Var_diag.flags.writeable = True
for i in range(N):
    if (Var_diag[i] < 10**-15):
        Var_diag[i] = 10**-15

scale = np.sqrt(Var_diag)
p = norm(loc = E, scale = scale).cdf(0.3)

plt.plot( th, p, label = "Expected values" )
plt.xlabel(r"$\theta$", size=20)
plt.ylabel(r'$p(\theta)$', size=20)
plt.title("Probability " + r'$Y(\theta) < 0.3 $', size=15)
plt.savefig("prob_2c.png")
plt.grid()
plt.legend()
plt.show()


print("Best choice for next theta: ", th[np.argmax( p )])

