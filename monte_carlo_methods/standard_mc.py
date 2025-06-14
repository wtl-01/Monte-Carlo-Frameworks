import numpy as np
from scipy import stats

def generate_normal_samples(num_samples: int):
    """
    Generate samples from a standard normal distribution using the Box-Muller transform.
    """
    U_1 = stats.uniform.rvs(size=num_samples)
    U_2 = stats.uniform.rvs(size=num_samples)

    samples = [
        np.sqrt(-2 * np.log(U_1)) * np.cos(2 * np.pi * U_2),
        np.sqrt(-2 * np.log(U_1)) * np.sin(2 * np.pi * U_2)
    ]
    
    return np.transpose(samples)

def calculate_distance(summary_observed, summary_simulated, n):
    """
    The distance is defined as the absolute difference normalized by the number of observations.
    """
    return (1/n) * abs(summary_observed - summary_simulated)


def abc_rejection_sampler(observed_data, prior_sampler, simulator, summary_statistic, distance_function, 
    target_posterior_size=1000, epsilon=0.01):
    """
    Approximate Bayesian Computation (ABC) rejection sampler.
    observed_data: The observed data for which we want to infer the posterior distribution.
    prior_sampler: A function that samples from the prior distribution.
    simulator: A function that simulates data given a parameter from the prior.
    summary_statistic: A function that computes a summary statistic from the data.
    distance_function: A function that computes the distance between two summary statistics.
    target_posterior_size: The desired size of the posterior sample.
    epsilon: The threshold for accepting a simulated parameter based on the distance to the observed summary statistic.

    """
    posterior_samples = []
    total_simulations = 0
    accepted_count = 0
    n_observed = len(observed_data)
    
    summary_observed = summary_statistic(observed_data)

    while accepted_count < target_posterior_size:
        candidate_parameter = prior_sampler()
        
        simulated_data = simulator(candidate_parameter, n_observed)
        
        summary_simulated = summary_statistic(simulated_data)
        
        distance = distance_function(summary_observed, summary_simulated)

        if distance <= epsilon:
            posterior_samples.append(candidate_parameter)
            accepted_count += 1
            
        total_simulations += 1

    return posterior_samples, total_simulations

def classical_MC_sampler(f_x_fxn, params, h_x_fxn, N=1000):
    init_sample = f_x_fxn.rvs(**params, size=N)
    return np.mean(h_x_fxn(init_sample))

def importance_sampler(f_x_fxn, f_params, g_x_fxn, g_params, h_x_fxn, N=1000):
    init_sample = g_x_fxn.rvs(**g_params, size=N)
    weights = f_x_fxn.pdf(init_sample, **f_params) / g_x_fxn.pdf(init_sample, **g_params)
    return np.mean(h_x_fxn(init_sample) * weights)

def autonormalized_importance_sampler(f_x_fxn, f_params, g_x_fxn, g_params, h_x_fxn, N=1000):
    init_sample = g_x_fxn.rvs(**g_params, size=N)
    log_weights = np.log(f_x_fxn.pdf(init_sample, **f_params))  - np.log(g_x_fxn.pdf(init_sample, **g_params))

    w_max = np.max(log_weights)
    weights = np.exp(log_weights - w_max)
    norm_weights = weights / np.sum(weights)
    return np.mean(h_x_fxn(init_sample) * norm_weights)

def antithetic_MC_sampler(f_x_fxn, f_params, f_center, h_x_fxn, N=1000):
    init_sample = f_x_fxn.rvs(**f_params, size=N//2)
    other_sample = 2*f_center - init_sample
    return np.mean(h_x_fxn(init_sample) + h_x_fxn(other_sample))/2

def control_variates_sampler(f_x_fxn, f_params, h_x_fxn, h_x_approx_fxn, h_x_mean, N=1000):
    init_sample = f_x_fxn.rvs(**f_params, size=N)
    h_x_sample = h_x_fxn(init_sample)
    h_x_approx_sample = h_x_approx_fxn(init_sample)
    mean_h_prev = np.mean(h_x_sample - h_x_approx_sample)
    return h_x_mean + mean_h_prev


