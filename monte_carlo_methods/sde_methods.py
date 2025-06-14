import numpy as np
import math

# Example of a univariate Geometric Brownian Motion (GBM) simulation using the Euler-Maruyama method.
def univariate_GBM_EM(initial_value, time_span, time_step, drift, volatility):
    """
    Simulates a 1D Geometric Brownian Motion (GBM) using the Euler-Maruyama method.

    initial_value: The initial value of the process (S_0).
    time_span: The total time duration for the simulation.
    time_step: The time increment for each step in the simulation.
    drift: The drift coefficient (mu) of the GBM.
    volatility: The volatility coefficient (sigma) of the GBM.
    """
    path = [initial_value]
    timestamps = [0.0]
    
    num_steps = int(time_span / time_step)

    for i in range(num_steps):
        Z = np.random.standard_normal() 

        # Calculate the next value using the Euler-Maruyama method
        # Formula is: S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
        next_value = path[-1] * (1 + drift * time_step + 
                                 volatility * math.sqrt(time_step) * Z)
        
        path.append(next_value)
        timestamps.append(timestamps[-1] + time_step)
        
    return np.array(timestamps), np.array(path)


# Multivariate Geometric Brownian Motion (GBM) simulation using the Euler-Maruyama method.
def multivariate_EM(initial_values, time_span, time_step, drift_fn, diffusion_fn):
    """
    Simulates a multivariate stochastic differential equation (SDE) using the Euler-Maruyama method.
    Assumes the number of dimensions equals the number of Wiener processes.

    initial_values (np.ndarray): A 1D array of initial values (d dimensions).
    time_span (float): The total time duration for the simulation.
    time_step (float): The time increment for each step.
    drift_fn (callable): Function a(x) that returns the (d,) drift vector.
    diffusion_fn (callable): Function C(x) that returns the (d, d) diffusion matrix.
    """
    num_dimensions = len(initial_values)
    num_steps = int(time_span / time_step)
    
    path = np.zeros((num_steps + 1, num_dimensions))
    path[0] = initial_values
    
    timestamps = np.linspace(0, time_span, num_steps + 1)
    sqrt_dt = np.sqrt(time_step)

    for i in range(1, num_steps + 1):
        x_prev = path[i - 1]
        
        C = diffusion_fn(x_prev)
        L = np.linalg.cholesky(C)
        Z = np.random.normal(loc=0.0, scale=1.0, size=num_dimensions)
        dW = Z * sqrt_dt
        path[i] = x_prev + drift_fn(x_prev) * time_step + L @ dW

    return timestamps, path

# Simplified Milstein Method for Any SDE


# General multivariate Milstein method for Any SDE
def multivariate_milstein_general(initial_values, time_span, time_step, drift_fn, volatility_fn, volatility_gradient_fn
):
    """
    Simulates a multivariate SDE using the general Milstein method for a
    general (non-diagonal) volatility matrix.

    Assumes the number of dimensions equals the number of Wiener processes.

    initial_values (np.ndarray): A 1D array of initial values (d dimensions).
    time_span (float): The total time duration for the simulation.
    time_step (float): The time increment for each step.
    drift_fn (callable): Function a(x) that returns the (d,) drift vector.
    volatility_fn (callable): Function B(x) that returns the (d, d) volatility matrix.
    volatility_gradient_fn (callable): Function grad_B(x) that returns the gradient of the volatility matrix, a d^3 tensor
    """
    num_dimensions = len(initial_values)
    num_steps = int(time_span / time_step)
    
    path = np.zeros((num_steps + 1, num_dimensions))
    path[0] = initial_values
    
    timestamps = np.linspace(0, time_span, num_steps + 1)
    sqrt_dt = np.sqrt(time_step)

    for i in range(1, num_steps + 1):
        x_prev = path[i-1]
        
        dW = np.random.normal(loc=0.0, scale=sqrt_dt, size=num_dimensions)
        
        a = drift_fn(x_prev)
        B = volatility_fn(x_prev)
        
        euler_part = a * time_step + B @ dW
        milstein_correction = np.zeros(num_dimensions)
        grad_B = volatility_gradient_fn(x_prev)
        
        for k in range(num_dimensions):
            sum_val = 0.0
            for j in range(num_dimensions):
                for l in range(num_dimensions):
                    dBk_dxl = grad_B[k, j, l]
                    
                    term = dBk_dxl * B[l, j] * (dW[j] * dW[l] - (time_step if j == l else 0.0))
                    sum_val += term
            milstein_correction[k] = 0.5 * sum_val

        path[i] = x_prev + euler_part + milstein_correction

    return timestamps, path
