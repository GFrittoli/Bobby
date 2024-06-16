''' File containing functions to compute derivatives of cosmological power spectra and related quantities.

Functions:
- polynomial_derivative: Computes the polynomial derivative of cosmological power spectra with respect to various cosmological parameters.
- finite_coeff_derivatives: Computes the derivatives for all parameters in cl_data using finite difference coefficients.
- richardson_extrapolation: Computes the derivatives for all parameters in cl_data using Richardson extrapolation.

'''







import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))

import numpy as np
import Derivate_coeff as dc
import concurrent.futures

import numpy as np
from numpy import polyfit, polyval
import os



from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


import Utilities as ut
import Fisher_Matrix as fm


# Ensure the rest of your functions follow here






def polynomial_derivative(cl_data, map, vars, fid_values, epsilon, scaling_factor=True):
    """
    Computes the polynomial derivative of cosmological power spectra with respect to various cosmological parameters.

    Args:
    cl_data (dict): Dictionary containing power spectra data for different variables.
    map (dict): Mapping of observational quantities to their respective indexes.
    vars (list): List of variables corresponding to the data in cl_data.
    fid_values (list): List of fiducial values for each variable.
    epsilon (list): List of perturbation factors applied to each fiducial value.
    scaling_factor (bool): Flag to determine whether to apply scaling for the CMB deflection-angle (to get the CMB lensing convergence) based on multipoles.

    Returns:
    tuple: Contains dictionaries for derivatives, R-squared values, scaled (convergence) and rescaled (without the ell*(ell+1) term) derivative data.
    """

    epsilon = np.array(epsilon)  # Ensure epsilon is a numpy array for calculations.
    # Initialize structures for output.
    key = list(cl_data.keys())[0]
    shape = cl_data[key].shape
    ells = shape[0]
    ls = np.arange(2, shape[0] + 2)
    normfactor = ls * (ls + 1) / (2 * np.pi)  # Normalization factor based on multipoles.
    probes = shape[1]
    fiducial_index = (shape[2] - 1) // 2  # Index for the central (fiducial) value in data arrays.
    derivatives_data = {var: np.zeros((ells, probes)) for var in cl_data.keys()}
    r_squared_data = {var: np.zeros((ells, probes)) for var in cl_data.keys()}

    # Process each variable to calculate derivatives and R-squared values.
    for var in cl_data.keys():
        print(f"Processing {var}...")
        data = cl_data[var]  # Extract data for current variable.
        fiducial_value = fid_values[vars.index(var)]  # Fiducial value for current variable.

        # Generate x values for polynomial fitting around the fiducial value.
        x = fiducial_value + epsilon * fiducial_value
        
        # Loop through each multipole and probe to fit a polynomial and calculate its derivative.
        for i in tqdm(range(ells)):
            for j in range(probes):
                y = data[i, j, :]
                coeffs = np.polyfit(x, y, 2)  # Fit a quadratic polynomial.
                pol = np.poly1d(coeffs)
                derivative_at_fiducial = pol.deriv()(fiducial_value)  # Derivative at fiducial value.

                # Calculate R^2 to assess fit quality.
                y_pred = pol(x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                derivatives_data[var][i, j] = derivative_at_fiducial
                r_squared_data[var][i, j] = r_squared

    # Scale derivatives if required.
    derivate_data_scaled = np.stack([derivatives_data[var] for var in vars], axis=2)
    if scaling_factor:
        for i, obs in enumerate(map.keys()):
            p_count = obs.count('P')
            if p_count > 0:
                scaling = (0.5 * np.sqrt(ls * (ls + 1))) ** p_count
                derivate_data_scaled[:, i, :] *= scaling.reshape(-1, 1)

    # Rescale the data using the normalization factor.
    derivate_data_rescaled = derivate_data_scaled / normfactor[:, np.newaxis, np.newaxis]
    
    return derivatives_data, r_squared_data, derivate_data_scaled, derivate_data_rescaled





def finite_coeff_derivatives(n_probes, cl_data, steps, map, vars, scaling=True):
    # Initialize output structures
    key = list(cl_data.keys())[0]
    ell = np.arange(cl_data[key].shape[0]) + 2  # Assuming ls starts at 2
    normfactor = (ell * (ell + 1)) / (2 * np.pi)
    
    derivatives = {}
    for var in tqdm(vars, desc="Computing derivatives for variables"):
        cl_param = cl_data[var]
        fiducial_index = (cl_param.shape[2] - 1) // 2
        n_points_derivative = cl_param.shape[2] - 1
        coeff = dc.coefficients(1, acc=n_points_derivative, offsets=None, symbolic=True, analytic_inv=False)
        coeff_center_list = coeff['center']['coefficients']
        coeff_list = [eval(str(c)) for c in coeff_center_list]

        # Compute the derivative for each probe
        Rcl_deriv = np.zeros((len(ell), n_probes))
        for jj in tqdm(range(n_probes), leave=False, desc=f"Derivatives for {var}"):
            for ii in range(len(coeff_list)):
                Rcl_deriv[:, jj] += coeff_list[ii] * cl_param[:, jj, fiducial_index + ii - len(coeff_list) // 2]
            Rcl_deriv[:, jj] /= steps[var] * normfactor

        derivatives[var] = Rcl_deriv

    # Combine all derivatives into a single array
    derivative_fin = np.stack([derivatives[var] for var in vars], axis=2)
    
    # Apply scaling if needed
    derivative_fin_scaled = np.copy(derivative_fin)
    if scaling:
        for i, obs in enumerate(tqdm(map, desc="Applying scaling")):
            p_count = obs.count('P')
            if p_count > 0:
                scaling_factor = (0.5 * np.sqrt(ell * (ell + 1))) ** p_count
                derivative_fin_scaled[:, i, :] *= scaling_factor.reshape(-1, 1)



    return derivative_fin, derivative_fin_scaled






def richardson_extrapolation(cl_data, epsilon, base_step, steps, map, scaling_factor=True):
    """
    Computes the derivatives for all parameters in cl_data using Richardson extrapolation.

    Args:
    cl_data (dict): Dictionary with each value being a numpy array of shape (n_data, n_quantities, n_perturbations).
    epsilon (numpy array): Array of perturbation values.
    base_step (float): The base step size.
    steps (dict): Dictionary of step sizes for each parameter.
    scaling_factor (bool): Flag to determine whether to apply scaling for the CMB deflection-angle (to get the CMB lensing convergence) based on multipoles.

    Returns:
    tuple: Dictionaries of derivatives, errors, scaled (convergence), and rescaled (without the ell*(ell+1) term) derivative data.
    """
    def find_optimal_richardson_indices(epsilon, base_step):
        epsilon_abs = np.abs(epsilon)
        # Filter out zero values and select steps that are powers of 2 times the base step
        valid_steps = [step for step in epsilon_abs if step != 0 and np.log2(step / base_step).is_integer()]
        valid_steps = sorted(set(valid_steps), reverse=True)
        
        indices = []
        for step in valid_steps:
            pos_index = np.where(epsilon == step)[0]
            neg_index = np.where(epsilon == -step)[0]
            if pos_index.size > 0 and neg_index.size > 0:
                indices.append(pos_index[0])
                indices.append(neg_index[0])
        
        return indices

    derivatives = {}
    errors = {}

    for param, data in cl_data.items():
        fiducial_value = steps[param] / base_step  # Extract the fiducial value
        selected_indices = find_optimal_richardson_indices(epsilon, base_step)
        ntab = len(selected_indices) // 2  # Halve the number of indices for effective steps
        
        n_data, n_quantities, _ = data.shape
        der = np.zeros((n_data, n_quantities))
        err = np.full((n_data, n_quantities), np.inf)

        diffs = {}
        for i in range(ntab):
            idx_plus = selected_indices[2*i]
            idx_minus = selected_indices[2*i+1]
            hs = epsilon[idx_plus]  # Use the actual epsilon value
            Dplus = data[:, :, idx_plus]
            Dminus = data[:, :, idx_minus]
            central_diff = (Dplus - Dminus) / (2 * fiducial_value * hs)  # Correctly divide by fiducial * epsilon value
            diffs[hs] = central_diff

        # Perform Richardson extrapolation
        for i in range(1, ntab):
            hs = epsilon[selected_indices[2*(i-1)]]  # step size used for current extrapolation
            for j in range(1, i + 1):
                fac = 2 ** (2 * j)
                diffs[hs] = (fac * diffs[hs/2] - diffs[hs]) / (fac - 1)
            
            # Update the derivative and error arrays based on improvements from the last extrapolation only
            new_err = np.abs(diffs[hs] - der)
            improved = new_err < err
            der[improved] = diffs[hs][improved]
            err[improved] = new_err[improved]

        derivatives[param] = der
        errors[param] = err

    # Get the shape from one of the data entries
    key = list(cl_data.keys())[0]
    shape = cl_data[key].shape
    ells = shape[0]
    ls = np.arange(2, shape[0] + 2)
    normfactor = ls * (ls + 1) / (2 * np.pi)  # Normalization factor based on multipoles.

    derivative_data_scaled = np.stack([derivatives[var] for var in vars], axis=2)

    if scaling_factor:
        for i, obs in enumerate(map.keys()):
            p_count = obs.count('P')
            if p_count > 0:
                scaling = (0.5 * np.sqrt(ls * (ls + 1))) ** p_count
                derivative_data_scaled[:, i, :] *= scaling.reshape(-1, 1)

    # Rescale the data using the normalization factor.
    derivative_data_rescaled = derivative_data_scaled / normfactor[:, np.newaxis, np.newaxis]
    
    return derivatives, errors, derivative_data_scaled, derivative_data_rescaled













