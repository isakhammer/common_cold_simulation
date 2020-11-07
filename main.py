import numpy as np
from stoch_mod_project_2 import P1
from stoch_mod_project_2 import P2

"""
Problem 1
"""
# c)
print("\n1c")
params = {"alpha": 0.1, "lambda": 100, "mu_L": 7, "mu_H": 20}
times, states = P1.simulate_process(365 * 5, params)
P1.plot_process(times, states)

# d) and e)
print("\n1d, 1e")
params = {"alpha": 0.1, "lambda": 100, "mu_L": 7, "mu_H": 20}
times, states = P1.simulate_process(365 * 10 ** 3, params)

"""
Problem 2
"""

# 5 evaluation points
print("\n2a, 2b")
t_0, t_1, E, n = 0.25, 0.50, 0.5, 51
t_A = np.linspace(t_0, t_1, n)
t_B = np.array([0.3, 0.35, 0.39, 0.41, 0.45])
y_B = np.array([0.5, 0.32, 0.40, 0.35, 0.60])
P2.generate_plots(E, t_A, t_B, y_B, "plot_1")

# 6 evaluation points
print("\n2c, 2c")
t_B = np.array([0.3, 0.33, 0.35, 0.39, 0.41, 0.45])
y_B = np.array([0.5, 0.4, 0.32, 0.40, 0.35, 0.60])
P2.generate_plots(E, t_A, t_B, y_B, "plot_2")
