import numpy as np
from scipy import stats

### Pseudocode for Gibbs Sampler

def gibbs_sampler_bivariate_normal(mean, cov, initial_position, num_samples=10000):
    """
    A Gibbs sampler for bivariate normal distributions.
    mean: A tuple or list of means for the two variables (mu1, mu2).
    cov: A 2x2 covariance matrix for the bivariate normal distribution.
    """

    mu1, mu2 = mean[0], mean[1]
    s1, s2 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    rho = cov[0, 1] / (s1 * s2)

    samples = np.zeros((num_samples, 2))
    samples[0] = initial_position

    x1, x2 = initial_position[0], initial_position[1]

    for i in range(1, num_samples):
        mean_cond_x1 = mu1 + rho * (s1 / s2) * (x2 - mu2)
        var_cond_x1 = (1 - rho**2) * s1**2
        x1 = np.random.normal(loc=mean_cond_x1, scale=np.sqrt(var_cond_x1))
        
        mean_cond_x2 = mu2 + rho * (s2 / s1) * (x1 - mu1)
        var_cond_x2 = (1 - rho**2) * s2**2
        x2 = np.random.normal(loc=mean_cond_x2, scale=np.sqrt(var_cond_x2))
        
        samples[i] = [x1, x2]

    return samples


### Markov Chain Monte Carlo (MCMC) Samplers

def mala_multivariate_sampler(log_target_pdf, log_target_gradient, initial_position,
    num_samples=10000, step_size=0.01, target_params=None):
    """
    N-dimensional (R^N) Metropolis-Adjusted Langevin Algorithm (MALA) sampler.

    log_target_pdf: Function that computes the log of the target distribution's PDF.
    log_target_gradient: Function that computes the gradient of the log target distribution.
    initial_position: Initial position for the sampler (array-like).
    num_samples: Number of samples to generate.
    step_size: Step size for the Langevin proposal.
    target_params: Optional parameters for the target distribution functions.
    """
    if target_params is None:
        target_params = {}

    dim = len(initial_position)
    
    proposal_cov = step_size * np.eye(dim)
    
    samples = np.zeros((num_samples, dim))
    samples[0] = initial_position
    current_pos = initial_position
    acceptance_count = 0

    for i in range(1, num_samples):
        grad_current = log_target_gradient(current_pos, **target_params)
        proposal_mean = current_pos + 0.5 * step_size * grad_current
        
        proposed_pos = stats.multivariate_normal.rvs(mean=proposal_mean, cov=proposal_cov)

        log_target_proposed = log_target_pdf(proposed_pos, **target_params)
        log_target_current = log_target_pdf(current_pos, **target_params)

        log_q_forward = stats.multivariate_normal.logpdf(proposed_pos, mean=proposal_mean, cov=proposal_cov)

        grad_proposed = log_target_gradient(proposed_pos, **target_params)
        reverse_proposal_mean = proposed_pos + 0.5 * step_size * grad_proposed
        log_q_reverse = stats.multivariate_normal.logpdf(current_pos, mean=reverse_proposal_mean, cov=proposal_cov)

        log_acceptance_ratio = (
            (log_target_proposed - log_target_current) +
            (log_q_reverse - log_q_forward)
        )
        
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current_pos = proposed_pos
            acceptance_count += 1
        
        samples[i] = current_pos

    acceptance_rate = acceptance_count / (num_samples - 1)
    return samples, acceptance_rate

def rwmh_multivariate_sampler(log_target_pdf, initial_position, proposal_cov,
    num_samples=10000,target_params=None):
    """
    R^N Random Walk Metropolis-Hastings (RWMH) sampler.
    log_target_pdf: Function that computes the log of the target distribution's PDF.
    initial_position: Initial position for the sampler (array-like).
    proposal_cov: Covariance matrix for the proposal distribution.
    num_samples: Number of samples to generate.
    target_params: Optional parameters for the target distribution functions.
    """
    if target_params is None:
        target_params = {}
        
    dim = len(initial_position)
    samples = np.zeros((num_samples, dim))
    samples[0] = initial_position
    current_pos = initial_position
    acceptance_count = 0

    for i in range(1, num_samples):
        proposed_pos = stats.multivariate_normal.rvs(mean=current_pos, cov=proposal_cov)

        log_target_proposed = log_target_pdf(proposed_pos, **target_params)
        log_target_current = log_target_pdf(current_pos, **target_params)
        log_acceptance_ratio = log_target_proposed - log_target_current

        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current_pos = proposed_pos
            acceptance_count += 1
        
        samples[i] = current_pos

    acceptance_rate = acceptance_count / (num_samples - 1)
    return samples, acceptance_rate

def ismh_multivariate_sampler(log_target_pdf, log_proposal_pdf, proposal_sampler, initial_position,
    num_samples=10000, target_params=None, proposal_params=None):
    """
    R^N Importance Sampling Metropolis-Hastings (ISMH) sampler.
    log_target_pdf: Function that computes the log of the target distribution's PDF.
    log_proposal_pdf: Function that computes the log of the proposal distribution's PDF.
    proposal_sampler: Function that samples from the proposal distribution.
    initial_position: Initial position for the sampler (array-like).
    num_samples: Number of samples to generate.
    target_params: Optional parameters for the target distribution functions.
    """
    if target_params is None: target_params = {}
    if proposal_params is None: proposal_params = {}

    dim = len(initial_position)
    samples = np.zeros((num_samples, dim))
    samples[0] = initial_position
    current_pos = initial_position
    acceptance_count = 0
    
    log_proposal_current = log_proposal_pdf(current_pos, **proposal_params)

    for i in range(1, num_samples):
        proposed_pos = proposal_sampler(**proposal_params)

        log_target_proposed = log_target_pdf(proposed_pos, **target_params)
        log_target_current = log_target_pdf(current_pos, **target_params)
        log_proposal_proposed = log_proposal_pdf(proposed_pos, **proposal_params)

        log_acceptance_ratio = ((log_target_proposed - log_target_current) +(log_proposal_current - log_proposal_proposed))

        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current_pos = proposed_pos
            log_proposal_current = log_proposal_proposed 
            acceptance_count += 1
        
        samples[i] = current_pos

    acceptance_rate = acceptance_count / (num_samples - 1)
    return samples, acceptance_rate

def simulated_annealing_optimizer(objective_fn, initial_position, temp_schedule, proposal_scale_schedule, max_iter=20000):
    """
    Simulated Annealing optimizer for minimizing a given objective function.
    objective_fn: Function to minimize.
    initial_position: Initial position in the search space (array-like).
    temp_schedule: Function that defines the temperature schedule T(k) for iteration k.
    proposal_scale_schedule: Function that defines the scale of the proposal distribution at iteration k.
    max_iter: Maximum number of iterations for the optimization.
    """
    dim = len(initial_position)
    
    current_pos = np.copy(initial_position)
    current_obj_val = objective_fn(current_pos)
    
    best_pos = np.copy(current_pos)
    best_obj_val = current_obj_val

    for k in range(max_iter):
        T = temp_schedule(k, max_iter)
        if T <= 1e-6:
            break
        
        scale = proposal_scale_schedule(k, max_iter)
        proposed_pos = current_pos + np.random.randn(dim) * scale
        
        proposed_obj_val = objective_fn(proposed_pos)
        
        delta_E = proposed_obj_val - current_obj_val
        
        if delta_E < 0 or np.random.uniform() < np.exp(-delta_E / T):
            current_pos = proposed_pos
            current_obj_val = proposed_obj_val

            if current_obj_val < best_obj_val:
                best_pos = current_pos
                best_obj_val = current_obj_val
                
    return best_pos, best_obj_val

def parallel_tempering_sampler(potential_energy, gradient, initial_states, temperatures,
    num_samples=1000000, swap_interval=10, step_size=0.1, adjacent=True,):
    """
    Parallel Tempering sampler for multivariate distributions.
    potential_energy: Function that computes the potential energy U(x) of the target distribution.
    gradient: Function that computes the gradient of the potential energy U(x).
    initial_states: List of initial states for each chain.
    temperatures: List of temperatures for each chain.
    num_samples: Number of samples to generate for each chain.
    """
    num_chains = len(temperatures)
    dim = len(initial_states[0])
    betas = 1.0 / temperatures

    current_states = [np.copy(s) for s in initial_states]
    all_chains = [np.zeros((num_samples, dim)) for _ in range(num_chains)]
    
    swap_attempts = {}
    swap_successes = {}

    for i in range(num_samples):
        for j in range(num_chains):
            current_pos = current_states[j]
            beta_j = betas[j]

            grad_log_p = -beta_j * gradient(current_pos)
            proposal_mean = current_pos + 0.5 * step_size * grad_log_p
            proposed_pos = stats.multivariate_normal.rvs(mean=proposal_mean, cov=step_size * np.eye(dim))
            
            log_p_ratio = -beta_j * (potential_energy(proposed_pos) - potential_energy(current_pos))
            
            grad_log_p_proposed = -beta_j * gradient(proposed_pos)
            reverse_proposal_mean = proposed_pos + 0.5 * step_size * grad_log_p_proposed
            
            log_q_forward = stats.multivariate_normal.logpdf(proposed_pos, mean=proposal_mean, cov=step_size * np.eye(dim))
            log_q_reverse = stats.multivariate_normal.logpdf(current_pos, mean=reverse_proposal_mean, cov=step_size * np.eye(dim))
            log_q_ratio = log_q_reverse - log_q_forward

            log_alpha = log_p_ratio + log_q_ratio

            if np.log(np.random.uniform()) < log_alpha:
                current_states[j] = proposed_pos
            
            all_chains[j][i, :] = current_states[j]

        if i > 0 and i % swap_interval == 0:
            if adjacent:
                pairs_to_swap = [(k, k + 1) for k in range(num_chains - 1)]
            else: #random
                idx1, idx2 = np.random.choice(num_chains, 2, replace=False)
                pairs_to_swap = [(idx1, idx2)]

            for idx1, idx2 in pairs_to_swap:
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                swap_attempts[(idx1, idx2)] = swap_attempts.get((idx1, idx2), 0) + 1
                beta1, beta2 = betas[idx1], betas[idx2]
                pos1, pos2 = current_states[idx1], current_states[idx2]
                U1 = potential_energy(pos1)
                U2 = potential_energy(pos2)
                
                log_alpha_swap = (beta1 - beta2) * (U2 - U1)

                if np.log(np.random.uniform()) < log_alpha_swap:
                    current_states[idx1], current_states[idx2] = pos2, pos1
                    swap_successes[(idx1, idx2)] = swap_successes.get((idx1, idx2), 0) + 1
    
    rates = {key: swap_successes.get(key, 0) / val for key, val in swap_attempts.items()}
    return all_chains, rates

def leapfrog_integrator(q, p, gradient, step_size, num_steps, mass_matrix_inv):
    q_new, p_new = np.copy(q), np.copy(p)
    p_new -= 0.5 * step_size * gradient(q_new)

    for i in range(num_steps - 1):
        q_new += step_size * (mass_matrix_inv @ p_new)
        p_new -= step_size * gradient(q_new)

    q_new += step_size * (mass_matrix_inv @ p_new)
    p_new -= 0.5 * step_size * gradient(q_new)
    
    return q_new, -p_new

def hmc_sampler(potential_energy, gradient, initial_position,
    num_samples=10000, mass_matrix=None, step_size=0.1, num_leapfrog_steps=10):
    """
    Hamiltonian Monte Carlo (HMC) sampler for multivariate distributions.
    potential_energy: Function that computes the potential energy U(x) of the target distribution.
    gradient: Function that computes the gradient of the potential energy U(x).
    initial_position: Initial position for the sampler (array-like).
    num_samples: Number of samples to generate.
    mass_matrix: Optional mass matrix for the kinetic energy term. If None, uses identity matrix.
    """
    dim = len(initial_position)
    
    if mass_matrix is None:
        mass_matrix = np.eye(dim)
    
    mass_matrix_inv = np.linalg.inv(mass_matrix)

    samples = np.zeros((num_samples, dim))
    samples[0] = initial_position
    current_q = np.copy(initial_position)
    acceptance_count = 0

    for i in range(1, num_samples):
        p_initial = stats.multivariate_normal.rvs(mean=np.zeros(dim), cov=mass_matrix)
        
        q_current, p_current = np.copy(current_q), np.copy(p_initial)

        q_proposed, p_proposed = leapfrog_integrator(q_current, p_current, gradient, step_size, num_leapfrog_steps, mass_matrix_inv)

        U_current = potential_energy(q_current)
        K_current = 0.5 * p_current.T @ mass_matrix_inv @ p_current
        H_current = U_current + K_current

        U_proposed = potential_energy(q_proposed)
        K_proposed = 0.5 * p_proposed.T @ mass_matrix_inv @ p_proposed
        H_proposed = U_proposed + K_proposed
        
        log_alpha = H_current - H_proposed
        
        if np.log(np.random.uniform()) < log_alpha:
            current_q = q_proposed
            acceptance_count += 1
        
        samples[i] = current_q

    acceptance_rate = acceptance_count / (num_samples - 1)
    return samples, acceptance_rate

### Particle Filters

class KalmanFilter:
    """
    A simple Kalman Filter implementation.
    F: State transition matrix
    B: Control input matrix
    H: Observation matrix
    Q: Process noise covariance
    R: Observation noise covariance
    x0: Initial state estimate
    P0: Initial error covariance
    """
    
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        
    def predict(self, u=None):
        if u is None:
            u = np.zeros(self.B.shape[1])
            
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


def bootstrap_particle_filter(N, initial_particles, observations, transition_fn, likelihood_fn, process_noise_cov):
    """
    Bootstrap Particle Filter (Sequential Importance Resampling).
    
    N: Number of particles
    initial_particles: Initial particles (N x dim_x)
    observations: Observations (T x dim_y)
    transition_fn: Function to compute the state transition (N x dim_x)
    likelihood_fn: Function to compute the likelihood of observations given particles (T x N)
    process_noise_cov: Process noise covariance (dim_x x dim_x)

    """
    dim_x = initial_particles.shape[1]
    num_T = len(observations)
    
    all_particles = np.zeros((num_T + 1, N, dim_x))
    all_particles[0] = initial_particles
    estimated_trajectory = np.zeros((num_T + 1, dim_x))
    estimated_trajectory[0] = np.mean(initial_particles, axis=0)
    
    weights = np.ones(N) / N

    for t in range(num_T):
        process_noise = np.random.multivariate_normal(np.zeros(dim_x), process_noise_cov, size=N)
        particles = transition_fn(all_particles[t]) + process_noise

        likelihoods = likelihood_fn(observations[t], particles)
        
        log_weights = np.log(likelihoods + 1e-300)  
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        weights /= np.sum(weights)
        
        indices = np.random.choice(N, size=N, p=weights)
        all_particles[t+1] = particles[indices]

        weights.fill(1.0 / N)
        estimated_trajectory[t+1] = np.mean(all_particles[t+1], axis=0)
        
    return estimated_trajectory, all_particles


def optimal_particle_filter(N, initial_particles, observations, F, B, H, Q, R, 
                            control_inputs=None):
    """
    Optimal Particle Filter for Linear Gaussian Systems.
    Uses the optimal proposal distribution p(x_t | x_{t-1}, y_t).
    N: Number of particles
    initial_particles: Initial particles (N x dim_x)
    observations: Observations (T x dim_y)
    F: State transition matrix (dim_x x dim_x)
    B: Control input matrix (dim_x x dim_u)
    H: Observation matrix (dim_y x dim_x)
    Q: Process noise covariance (dim_x x dim_x)
    R: Observation noise covariance (dim_y x dim_y)
    control_inputs: Control inputs (T x dim_u), optional

    """
    dim_x = F.shape[0]
    dim_y = H.shape[0]
    num_T = len(observations)
    
    if control_inputs is None:
        control_inputs = np.zeros((num_T, B.shape[1]))
        
    all_particles = np.zeros((num_T + 1, N, dim_x))
    all_particles[0] = initial_particles
    estimated_trajectory = np.zeros((num_T + 1, dim_x))
    estimated_trajectory[0] = np.mean(initial_particles, axis=0)
    
    weights = np.ones(N) / N
    Q_inv = np.linalg.inv(Q)
    R_inv = np.linalg.inv(R)
    Sigma_t_inv = Q_inv + H.T @ R_inv @ H
    Sigma_t = np.linalg.inv(Sigma_t_inv)
    
    for t in range(num_T):
        log_weights = np.zeros(N)
        particles = np.zeros((N, dim_x))
        
        for i in range(N):
            x_prev = all_particles[t, i]
            
            x_pred = F @ x_prev + B @ control_inputs[t]
            proposal_mean = Sigma_t @ (Q_inv @ x_pred + H.T @ R_inv @ observations[t])
            particles[i] = np.random.multivariate_normal(proposal_mean, Sigma_t)

            y_mean = H @ particles[i]
            log_p_yt_given_xt = stats.multivariate_normal.logpdf(observations[t], y_mean, R)
            log_p_xt_given_xt_1 = stats.multivariate_normal.logpdf(particles[i], x_pred, Q)
            log_q_xt = stats.multivariate_normal.logpdf(particles[i], proposal_mean, Sigma_t)
            log_weights[i] = np.log(weights[i]) + log_p_yt_given_xt + \
                             log_p_xt_given_xt_1 - log_q_xt
        
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        weights /= np.sum(weights)
        ess = 1.0 / np.sum(weights**2)

        if ess < N / 2:
            indices = np.random.choice(N, size=N, p=weights)
            all_particles[t+1] = particles[indices]
            weights.fill(1.0 / N)
        else:
            all_particles[t+1] = particles
        
        estimated_trajectory[t+1] = np.sum(weights[:, np.newaxis] * all_particles[t+1], axis=0)
        
    return estimated_trajectory, all_particles


def auxiliary_particle_filter(N, initial_particles, observations, F, B, H, Q, R, 
                              control_inputs=None):
    """
    Auxiliary Particle Filter (APF) for improved performance.

    N: Number of particles
    initial_particles: Initial particles (N x dim_x)
    observations: Observations (T x dim_y)
    F: State transition matrix (dim_x x dim_x)
    B: Control input matrix (dim_x x dim_u)
    H: Observation matrix (dim_y x dim_x)
    Q: Process noise covariance (dim_x x dim_x)
    R: Observation noise covariance (dim_y x dim_y)
    control_inputs: Control inputs (T x dim_u), optional
    """
    dim_x = F.shape[0]
    num_T = len(observations)
    
    if control_inputs is None:
        control_inputs = np.zeros((num_T, B.shape[1]))
        
    all_particles = np.zeros((num_T + 1, N, dim_x))
    all_particles[0] = initial_particles
    estimated_trajectory = np.zeros((num_T + 1, dim_x))
    estimated_trajectory[0] = np.mean(initial_particles, axis=0)
    
    weights = np.ones(N) / N

    for t in range(num_T):
        predicted_particles = F @ all_particles[t].T + B @ control_inputs[t:t+1].T
        predicted_particles = predicted_particles.T
        predicted_measurements = H @ predicted_particles.T
        auxiliary_weights = np.zeros(N)
        
        for i in range(N):
            auxiliary_weights[i] = weights[i] * stats.multivariate_normal.pdf(observations[t], predicted_measurements[:, i], R)
        
        auxiliary_weights /= np.sum(auxiliary_weights)
        indices = np.random.choice(N, size=N, p=auxiliary_weights)
        
        resampled_particles = all_particles[t, indices]
        process_noise = np.random.multivariate_normal(np.zeros(dim_x), Q, size=N)
        particles = F @ resampled_particles.T + B @ control_inputs[t:t+1].T
        particles = particles.T + process_noise
        
        measurements = H @ particles.T
        log_weights = np.zeros(N)
        
        for i in range(N):
            log_likelihood = stats.multivariate_normal.logpdf(observations[t], measurements[:, i], R)
            
            parent_idx = indices[i]
            log_aux_weight = np.log(auxiliary_weights[parent_idx] + 1e-300)
            log_weights[i] = log_likelihood - log_aux_weight
        
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        weights /= np.sum(weights)
        
        all_particles[t+1] = particles
        estimated_trajectory[t+1] = np.sum(weights[:, np.newaxis] * particles, axis=0)
        
    return estimated_trajectory, all_particles