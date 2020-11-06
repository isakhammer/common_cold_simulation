import numpy as np
import matplotlib.pyplot as plt


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
    print(H_b)

    Sigma_b_inv = np.linalg.inv(Sigma_b)

    E_a_b = mu_a + Sigma_ab @ Sigma_b_inv @ (x_b - mu_b)
    var_a_b = Sigma_a - Sigma_ab @ Sigma_b_inv @ Sigma_ab.T
    return E_a_b, var_a_b


# Grid
t_0 = 0.25
t_1 = 0.50
n = 51
t_a = np.linspace(t_0, t_1, n).reshape(n, 1)
y_b = np.array([0.5, 0.32, 0.40, 0.35, 0.60]).reshape(5, 1)
t_b = np.array([0.3, 0.35, 0.39, 0.41, 0.45]).reshape(5, 1)
mu_a = 0.5 * np.ones((51, 1))
mu_b = 0.5 * np.ones((5, 1))
E, Var = gauss_process(t_a, mu_a, mu_b, t_b, y_b)
print(E.shape, Var.shape)
plt.scatter(t_b, y_b, label="Data")
plt.plot(t_a, E, label="Expected values" )
plt.legend()
plt.show()
