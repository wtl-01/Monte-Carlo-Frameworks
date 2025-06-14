import numpy as np
from scipy import stats

def importance_sampler(f_fxn, f_params, g_fxn, g_params, h_fxn, norm=True, N=1000000):
    """
    Performs importance sampling to estimate the expectation of h(x).

    f_fxn: target distribution (pdf)
    f_params: parameters for the target distribution
    g_fxn: proposal distribution (pdf)
    g_params: parameters for the proposal distribution
    h_fxn: function to evaluate the expectation E[h(x)]
    """
    samples = g_fxn.rvs(**g_params, size=N)
    weights = f_fxn.pdf(samples, **f_params) / g_fxn.pdf(samples, **g_params)
    sum_of_weights_squared = np.sum(weights**2)
    if sum_of_weights_squared == 0:
        ESS = 0
    else:
        ESS = np.sum(weights)**2 / sum_of_weights_squared

    if norm:
        weights /= np.sum(weights)

    estimated_expectation = np.mean(h_fxn(samples) * weights)

    return estimated_expectation, ESS

def rejection_sampler(f_fxn, f_params, g_fxn, g_params, M=2.5, N=1000000):
    """
    Rejection Sampling in R^1
    f_fxn: target distribution (pdf)
    f_params: parameters for the target distribution
    g_fxn: proposal distribution (pdf)
    g_params: parameters for the proposal distribution
    M: constant such that f(x) <= M * g(x) for all x
    N: number of samples to generate
    """
    accepted = []
    num_accepted = 0
    num_total = 0
    while num_accepted < N:
        init_sample = g_fxn.rvs(**g_params, size=1)
        accprob = f_fxn.pdf(init_sample, **f_params) / (M * g_fxn.pdf(init_sample, **g_params))
        u_sample = np.random.uniform(0, 1, size=1)
        if u_sample <= accprob:
            accepted.append(init_sample)
            num_accepted += 1
        num_total += 1

    return accepted, num_total

def ismh_sampler(f_fxn, f_params, q_fxn, q_params, N=1000000):
    """
    Independence Sampler based Metropolis Hastings in R^1
    f_fxn: target distribution (pdf)
    f_params: parameters for the target distribution
    q_fxn: proposal distribution (pdf)
    q_params: parameters for the proposal distribution
    N: number of samples to generate
    """
    init_sample = q_fxn.rvs(**q_params, size=1)
    accepted = [init_sample]
    num_accepted = 0
    num_total = 0
    num_acc_triv = 0

    prev_sample = init_sample

    while num_accepted < N:
        new_sample = q_fxn.rvs(**q_params, size=1)
        fz_over_fx = f_fxn.pdf(new_sample, **f_params) / f_fxn.pdf(prev_sample, **f_params)
        qx_over_qz = q_fxn.pdf(prev_sample, **q_params) / q_fxn.pdf(new_sample, **q_params)
        alpha = min(1, fz_over_fx * qx_over_qz)

        u_sample = np.random.uniform(0, 1, size=1)
        if u_sample <= alpha:
            accepted.append(new_sample)
            num_accepted += 1
            prev_sample = new_sample
            num_acc_triv += 1
        else:
            accepted.append(prev_sample)
            num_accepted += 1
        num_total += 1

    return accepted, num_acc_triv

def rwmh_sampler(f_fxn, f_params, q_fxn, q_params, N=10000):
    """
    RWMH based Metropolis Hastings in R^1
    f_fxn: target distribution (pdf)
    f_params: parameters for the target distribution
    q_fxn: proposal distribution (pdf)
    q_params: parameters for the proposal distribution
    N: number of samples to generate
    """
    init_sample = q_fxn.rvs(**q_params, size=1)
    accepted = [init_sample]
    num_accepted = 0
    num_total = 0
    num_acc_triv = 0

    prev_sample = init_sample
    while num_accepted < N:
        new_sample = prev_sample + q_fxn.rvs(**q_params, size=1)
        fz_over_fx = f_fxn.pdf(new_sample, **f_params) / f_fxn.pdf(prev_sample, **f_params)
        alpha = min(1, fz_over_fx)

        u_sample = np.random.uniform(0, 1, size=1)
        if u_sample <= alpha:
            accepted.append(new_sample)
            num_accepted += 1
            prev_sample = new_sample
            num_acc_triv += 1
        else:
            accepted.append(prev_sample)
            num_accepted += 1
        num_total += 1

    return accepted, num_acc_triv

def mala_sampler(log_target_pdf, log_target_gradient, initial_position,
    num_samples=10000, step_size=0.1, target_params=None):
    """
    R^1 Metropolis-Adjusted Langevin Algorithm (MALA) sampler.

    log_target_pdf: Function that computes the log of the target distribution's PDF.
    log_target_gradient: Function that computes the gradient of the log target distribution.
    initial_position: Initial position for the sampler.
    num_samples: Number of samples to generate.
    step_size: Step size for the Langevin proposal.
    target_params: Optional parameters for the target distribution functions.
    """
    if target_params is None:
        target_params = {}

    samples = np.zeros(num_samples)
    samples[0] = initial_position
    current_pos = initial_position
    acceptance_count = 0

    for i in range(1, num_samples):
        grad_current = log_target_gradient(current_pos, **target_params)
        proposal_mean = current_pos + 0.5 * step_size * grad_current
        proposed_pos = np.random.normal(loc=proposal_mean, scale=np.sqrt(step_size))

        log_target_proposed = log_target_pdf(proposed_pos, **target_params)
        log_target_current = log_target_pdf(current_pos, **target_params)

        log_q_forward = stats.norm.logpdf(proposed_pos, loc=proposal_mean, scale=np.sqrt(step_size))


        grad_proposed = log_target_gradient(proposed_pos, **target_params)
        reverse_proposal_mean = proposed_pos + 0.5 * step_size * grad_proposed
        log_q_reverse = stats.norm.logpdf(current_pos, loc=reverse_proposal_mean, scale=np.sqrt(step_size))

        log_acceptance_ratio = ((log_target_proposed - log_target_current) + (log_q_reverse - log_q_forward))
        
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current_pos = proposed_pos
            acceptance_count += 1
        
        samples[i] = current_pos

    acceptance_rate = acceptance_count / (num_samples - 1)
    return samples, acceptance_rate

