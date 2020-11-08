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
    Sigma_AcB[Sigma_AcB < 0] = 10**(-15)
    return mu_AcB, Sigma_AcB


def generate_plots(E, t_A, t_B, y_B, filename):
    mu_A, mu_B = E * np.ones_like(t_A), E * np.ones_like(t_B)
    mu_AcB, Sigma_AcB = gauss_process(mu_A, mu_B, t_A, t_B, y_B, matern_15)
    var_A = np.diagonal(Sigma_AcB)
    mu_AcB_l = mu_AcB - np.sqrt(var_A) * 1.645
    mu_AcB_u = mu_AcB + np.sqrt(var_A) * 1.645
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.scatter(t_B, y_B, label="Evaluation points", color='black')
    ax0.plot(t_A, mu_AcB, label=r'$\mu_{A \mid B}$')
    ax0.fill_between(t_A, mu_AcB_l, mu_AcB_u, color="gray", alpha=0.5)
    ax0.set_xlabel(r'$\theta$', size=20)
    ax0.set_ylabel(r'$y(\theta)$', size=20)
    ax0.set_title(r'$\mathbf{\mu}_{A \mid B}$' + " and 90% prediction interval of " r'$y(\theta)$', size=15)
    ax0.grid()
    ax0.legend(prop={"size": 15})

    ###
    p = norm(loc=mu_AcB, scale=np.sqrt(var_A)).cdf(0.3)
    print("opt theta_", round(t_A[np.argmax(p)], 2))
    ax1.plot(t_A, p)
    ax1.set_xlabel(r"$\theta$", size=20)
    ax1.set_ylabel(r'$\mathrm{Pr}\{\theta < 0.3\}$', size=20)
    ax1.set_title(r'$\mathrm{Pr}\{\theta < 0.3\}$', size=20)
    plt.savefig(filename)
    ax1.grid()
    plt.show()



