import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_process(T, params):
    infections = T//10
    alpha, lam, mu_L, mu_H = params["alpha"], params["lambda"], params["mu_L"], params["mu_H"]
    states = np.zeros(2 * infections)
    infection_indicies = np.arange(1, 2 * infections, 2)
    states[infection_indicies] = 1
    # I_H = Heavy infection
    I_H_count = np.random.binomial(infections, alpha)  # I_H_count = total number of heavy infections
    I_H_indicies = np.random.choice(infection_indicies, I_H_count, replace=False)  # chronological order of heavy infections
    states[I_H_indicies] = 2
    sojourn_times = np.empty(2*infections)
    sojourn_times[infection_indicies - 1] = np.random.exponential(lam, infections)
    sojourn_times[infection_indicies] = np.random.exponential(mu_L, infections)
    sojourn_times[I_H_indicies] = np.random.exponential(mu_H, I_H_count)

    times = np.cumsum(sojourn_times)
    sojourn_times1 = sojourn_times[times < T]
    states1 = states[times < T]
    times1 = times[times < T]

    sojourn_times1 = np.append(sojourn_times1, T - times1[-1])
    times1 = np.append(times1, T)
    states1 = np.append(states1, states[len(states1)])

    func = np.vectorize(lambda state: np.sum(sojourn_times1[states1 == state]))
    state_space = np.array([0, 1, 2])
    long_run_mean_fractions_vec = func(state_space)
    long_run_mean_fractions_vec /= np.sum(long_run_mean_fractions_vec)
    print("estimated long run mean fraction vector: ", long_run_mean_fractions_vec)
    print("estimated long run mean infected fraction: ", 1 - long_run_mean_fractions_vec[0])
    indicies = np.arange(0, len(states1), 1)
    heavy_infection_ends = (times1[indicies[states1 == 2]])[:-1]
    heavy_infection_starts = (times1[indicies[states1 == 2] - 1])[1:]
    w_values = heavy_infection_starts - heavy_infection_ends
    if len(w_values != 0):
        print("mean time between heavy infections:", np.mean(w_values))
    else:
        print("less than two heavy infections")
    return times1, states1


def plot_process(times, states):
    times_to_plot = np.delete(np.append(np.array([0]), np.repeat(times, 2)), -1)
    states_to_plot = np.repeat(states, 2)
    fig, ax = plt.subplots()
    fig = plt.subplots_adjust(left=0.15, bottom=0.2)
    ax.set_title("State of individual over 5 years", size=30)
    ax.set_xlabel("Year", size=40)
    ax.set_ylabel("State", size=40)
    years = times[-1]//365 + 1
    ax.set_xticks(np.arange(0, years * 365 + 1, 365))
    ax.set_xticklabels(np.arange(0, years + 1, 1).astype(int))
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r'$S$', r'$I_L$', r'$I_H$'])
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.plot(times_to_plot, states_to_plot)
    plt.show()


def process_2(T, params):
    infections = T // 10
    lam, mu = params["lambda"], params["mu"]
    states = np.zeros(2 * infections)
    infection_indicies = np.arange(1, 2 * infections, 2)
    states[infection_indicies] = 1
    sojourn_times = np.empty(2 * infections)
    sojourn_times[infection_indicies - 1] = np.random.exponential(lam, infections)
    sojourn_times[infection_indicies] = np.random.exponential(mu, infections)

    times = np.cumsum(sojourn_times)
    sojourn_times1 = sojourn_times[times < T]
    states1 = states[times < T]
    times1 = times[times < T]

    sojourn_times1 = np.append(sojourn_times1, T - times1[-1])
    times1 = np.append(times1, T)
    states1 = np.append(states1, states[len(states1)])

    func = np.vectorize(lambda state: np.sum(sojourn_times1[states1 == state]))
    state_space = np.array([0, 1, 2])
    long_run_mean_fractions_vec = func(state_space)
    long_run_mean_fractions_vec /= np.sum(long_run_mean_fractions_vec)
    print("estimated long run mean fraction vector: ", long_run_mean_fractions_vec)
    print("estimated long run mean infected fraction: ", 1 - long_run_mean_fractions_vec[0])
    indicies = np.arange(0, len(states1), 1)
    heavy_infection_ends = (times1[indicies[states1 == 2]])[:-1]
    heavy_infection_starts = (times1[indicies[states1 == 2] - 1])[1:]
    w_values = heavy_infection_starts - heavy_infection_ends
    if len(w_values != 0):
        print("mean time between heavy infections:", np.mean(w_values))
    else:
        print("less than two heavy infections")
    return times1, states1
