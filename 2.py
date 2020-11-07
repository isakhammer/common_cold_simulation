import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial import distance_matrix as dm

def transpose_1d_arrs(arrs):
    return [arr.reshape(len(arr), 1) for arr in arrs]


def untranspose_1d_arrs(arrs):
    return [arr[:, 0] for arr in arrs]


def matern_15(d):
    return (1 + 15*d)*np.exp(-15*d)


def gauss_process(mu_A, mu_B, t_A, t_B, y_B, corr=matern_15, Var=0.5**2):
    mu_A, mu_B, t_A, t_B, y_B = transpose_1d_arrs([mu_A, mu_B, t_A, t_B, y_B])
    H_list = [dm(t[0], t[1], p=1) for t in [(t_A, t_A), (t_B, t_B), (t_A, t_B)]]
    Sigma_A, Sigma_B, Sigma_AB = [Var * corr(H) for H in H_list]
    Sigma_B_inv = np.linalg.inv(Sigma_B)
    mu_AcB = (mu_A + Sigma_AB @ Sigma_B_inv @ (y_B - mu_B))[:, 0]
    Sigma_AcB = Sigma_A - Sigma_AB @ Sigma_B_inv @ Sigma_AB.T
    return mu_AcB, Sigma_AcB


def generate_plots(E, t_A, t_B, y_B):
    mu_A, mu_B = E * np.ones_like(t_A), E * np.ones_like(t_B)
    mu_AcB, Sigma_AcB = gauss_process(mu_A, mu_B, t_A, t_B, y_B, matern_15)
    var_A = np.diagonal(Sigma_AcB)
    mu_AcB_l = mu_AcB - np.sqrt(var_A) * 1.645
    mu_AcB_u = mu_AcB + np.sqrt(var_A) * 1.645
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.scatter(t_B, y_B, label="Data")
    ax0.plot(t_A, mu_AcB, label=r'$\mu_{A \mid B}$')
    ax0.fill_between(t_A, mu_AcB_l, mu_AcB_u, color="gray", alpha=0.5)
    ax0.set_xlabel(r'$\theta$', size=20)
    ax0.set_ylabel(r'$y(\theta)$', size=20)
    ax0.set_title("Mean vector and 90% Prediction interval", size=15)
    #ax0.savefig("gp.png")
    ax0.grid()
    ax0.legend()

    p = norm(loc=mu_AcB, scale=np.sqrt(var_A)).cdf(0.3)

    ax1.plot(t_A, p, label="Expected values")
    ax1.set_xlabel(r"$\theta$", size=20)
    ax1.set_ylabel(r'$p(\theta)$', size=20)
    ax1.set_title("Probability " + r'$Y(\theta) < 0.3 $', size=15)
    #ax1.savefig("prob_2.png")
    ax1.grid()
    ax1.legend()
    plt.show()

# 5 evaluation points
t_0, t_1, E, n = 0.25, 0.50, 0.5, 51
t_A = np.linspace(t_0, t_1, n)
t_B = np.array([0.3, 0.35, 0.39, 0.41, 0.45])
y_B = np.array([0.5, 0.32, 0.40, 0.35, 0.60])
generate_plots(E, t_A, t_B, y_B)

# 6 evaluation points
t_A = np.linspace(t_0, t_1, n).reshape(n, 1)
t_B = np.array([0.3, 0.33, 0.35, 0.39, 0.41, 0.45]).reshape(6, 1)
y_B = np.array([0.5, 0.4, 0.32, 0.40, 0.35, 0.60]).reshape(6, 1)
generate_plots(E, t_A, t_B, y_B)



