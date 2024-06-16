'''File containing functions to compute the Fisher matrix and related quantities for cosmological analyses. It is specificcaly designed for Angular Power Spectra

Functions:

- compute_fisher_analysis: Compute the Fisher analysis based on the provided configuration.
- compute_fskies: Computes the fskies dictionary based on input parameters and a choice to set cross fsky values to the minimum of the two fields' fsky values.
- sigma: Define the covariance matrix for the Gaussian case.
- get_masked_sigma: Mask the covariance matrix for the Gaussian case in certain ranges of multipoles.
- inv_sigma: Invert the covariance matrix of the Gaussian case.
- get_masked_derivates: Apply the mask to the derivative of the power spectra.
- get_fisher: Compute the Fisher matrix, covariance, and errors.

'''









# Show plots inline, and load main getdist plot module and samples class
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))

import numpy as np
import Derivate_coeff as dc
import concurrent.futures

import numpy as np
from numpy import polyfit, polyval
import os
try:
    from tqdm.notebook import tqdm  # This is specifically for Jupyter notebooks
except ImportError:
    from tqdm import tqdm  # Standard console version as a fallback

import Utilities as ut

import Derivatives as dv



# Ensure the rest of your functions follow here


# Class to encapsulate the configuration parameters for the Fisher analysis
class FisherAnalysisConfig:
    def __init__(self, file_path, fields, vars, epsilon, fid_values, noisedict, base_path, directory_structure, fskies, lmax, lmin, excluded_probes_euclid, lminimi, lmassimi, fsky=None):
        """
        Initialize the FisherAnalysisConfig class with the necessary parameters.

        Parameters:
        - file_path: Path to the fiducial.txt file.
        - fields: List of fields used for the analysis.
        - vars: List of variables for the analysis.
        - epsilon: List of perturbation sizes for derivatives.
        - fid_values: List of fiducial values for the parameters.
        - noisedict: Dictionary containing noise information.
        - base_path: Base path to the data files.
        - directory_structure: Directory structure template for data files.
        - fskies: List of sky fractions for each field.
        - lmax: Maximum multipole moment.
        - lmin: Minimum multipole moment.
        - excluded_probes_euclid: List of probes to be excluded from the Euclid analysis.
        - lminimi: List of minimum multipole moments for each field.
        - lmassimi: List of maximum multipole moments for each field.
        - fsky: Optional sky fraction (default is None).
        """
        self.file_path = file_path
        self.fields = fields
        self.vars = vars
        self.epsilon = epsilon
        self.fid_values = fid_values
        self.noisedict = noisedict
        self.base_path = base_path
        self.directory_structure = directory_structure
        self.fskies = fskies
        self.lmax = lmax
        self.lmin = lmin
        self.excluded_probes_euclid = excluded_probes_euclid
        self.lminimi = lminimi
        self.lmassimi = lmassimi
        self.fsky = fsky

# Function to compute the Fisher analysis using the configuration parameters
def compute_fisher_analysis(config: FisherAnalysisConfig):
    """
    Compute the Fisher analysis based on the provided configuration.

    Parameters:
    - config: An instance of FisherAnalysisConfig containing the analysis parameters.

    Returns:
    - fisher: Fisher matrix.
    - covariance_fish_euclid: Covariance matrix from the Fisher analysis.
    - errori_euclid: Errors from the Fisher analysis.
    """
    # Step 1: Get column mapping
    n_fields = len(config.fields)
    n_probes = (n_fields * (n_fields + 1)) // 2
    n_params = len(config.vars)
    n_counts = ut.count_ws(config.fields)
    ls = np.arange(2, config.lmax + 1, 1)
    ls2 = np.insert(ls, [0, 0], [0, 0])

    # Get the column mapping for the fiducial data
    column_mapping = ut.get_mapping(config.file_path, n_fields)

    # Step 2: Get keys and indices
    keys = ut.get_keys(config.fields)
    indici = ut.get_Gauss_keys_v3(config.fields)

    # Step 3: Extract fiducial Cl from CAMB
    path_unlens = f'{config.base_path}/{config.directory_structure}/fiducial/unlensed/fiducial.txt'
    path_lens = f'{config.base_path}/{config.directory_structure}/fiducial/lensed/fiducial.txt'
    clsdict = ut.extract_fiducial_cl_from_CAMB(path_unlens, path_lens, column_mapping, n_counts, ls)
    directory_structure_var = config.directory_structure + '/{var}'

    # Step 4: Process the Cl data for the specified variables and epsilon values
    cl_data = ut.process_cl_data(config.vars, config.epsilon, column_mapping, ls, config.base_path, directory_structure_var)

    # Step 5: Compute derivatives data
    derivatives_data, r_squared_data, derivate_data_scaled, derivate_data_rescaled = dv.polynomial_derivative(
        cl_data, column_mapping, config.vars, config.fid_values, config.epsilon, scaling_factor=True)

    # Step 6: Compute fskies
    fskies = compute_fskies(n_fields, config.fields, config.fskies, use_min_fsky_for_crosses=True)

    # Step 7: Build the covariance matrix
    covariance = sigma(n_fields, config.lmax, config.lmin, config.fsky, fskies, indici, clsdict, config.noisedict)
    for i in range(config.lmin, len(ls2)):
        covariance[:,:,i] = (covariance[:,:,i]) / (2 * ls2 + 1)[i]

    # Step 8: Apply mask to the covariance
    masked_euclid = get_masked_sigma(n_fields, config.lmin, config.lmax, indici, covariance, config.excluded_probes_euclid, lmins=config.lminimi, lmaxs=config.lmassimi)
    inversa_euclid = inv_sigma(config.lmin, config.lmax, masked_euclid)
    masked_deriv_euclid = get_masked_derivates(n_fields, masked_sigma=masked_euclid, derivative=derivate_data_rescaled, ells=ls)

    # Step 9: Compute Fisher matrix, covariance and errors
    fisher, covariance_fish_euclid, errori_euclid = get_fisher(ls, n_params, masked_deriv_euclid, inversa_euclid)

    return fisher, covariance_fish_euclid, errori_euclid











def find_spectrum(input_dict, lmax, lmin, key):
    """Find a spectrum in a given dictionary.

    Returns the corresponding power spectrum for a given key. If the key is not found, it will try to find the reverse key. Otherwise, it will fill the array with zeros.

    Parameters:
        input_dict (dict):
            Dictionary where you want to search for keys.
        lmax (int):
            Maximum l value.
        lmin (int):
            Minimum l value.
        key (str):
            Key to search for.
    """
    # Create a zero array
    res = np.zeros(lmax + 1)

    # Try to find the key in the dictionary
    if key in input_dict:
        cov = input_dict[key]
    else:
        # Create the reverse key considering symmetric case 'AxB' -> 'BxA'
        parts = key.split('x')
        if len(parts) == 2:
            reverse_key = 'x'.join(parts[::-1])
            cov = input_dict.get(reverse_key, np.zeros(lmax + 1))
        else:
            # If the key does not contain 'x', use the original logic
            cov = input_dict.get(key[::-1], np.zeros(lmax + 1))

    # Fill the array with the requested spectrum
    res[lmin : lmax + 1] = cov[lmin : lmax + 1]

    return res




def sigma(n,lmax,lmin,fsky,fskies, keys, fiduDICT, noiseDICT):
        """Define the covariance matrix for the Gaussian case.

        In case of Gaussian likelihood, this returns the covariance matrix needed for the computation of the chi2. Note that the inversion is done in a separate funciton.

        Parameters:
            keys (dict):
                Keys for the covariance elements.

            fiduDICT (dict):
                Dictionary with the fiducial spectra.

            noiseDICT (dict):
                Dictionary with the noise spectra.
        """
        # The covariance matrix has to be symmetric.
        # The number of parameters in the likelihood is n.
        # The covariance matrix is a (n x n x lmax+1) ndarray.
        # We will store the covariance matrix in a (n x n x lmax+1) ndarray,
        # where n = int(n * (n + 1) / 2).
        n = int(n * (n + 1) / 2)
        res = np.zeros((n, n, lmax + 1))
        for i in range(n):  # Loop over all combinations of pairs of spectra
            for j in range(i, n):
                print(i,j,keys[i, j, :])
                #print(fskies[AB])
                C_AC = find_spectrum(
                    fiduDICT,lmax,lmin, keys[i, j, 0] +'x'+ keys[i, j, 2]
                )  # Find the fiducial spectra for each pair
                C_BD = find_spectrum(fiduDICT,lmax,lmin, keys[i, j, 1] +'x'+ keys[i, j, 3])
                C_AD = find_spectrum(fiduDICT,lmax,lmin, keys[i, j, 0] +'x'+ keys[i, j, 3])
                C_BC = find_spectrum(fiduDICT,lmax,lmin, keys[i, j, 1] +'x'+ keys[i, j, 2])
                N_AC = find_spectrum(noiseDICT,lmax,lmin, keys[i, j, 0] +'x'+ keys[i, j, 2])
                # Find the noise spectra for each pair
                N_BD = find_spectrum(noiseDICT,lmax,lmin, keys[i, j, 1] +'x'+ keys[i, j, 3])
                N_AD = find_spectrum(noiseDICT,lmax,lmin, keys[i, j, 0] +'x'+ keys[i, j, 3])
                N_BC = find_spectrum(noiseDICT,lmax,lmin, keys[i, j, 1] +'x'+ keys[i, j, 2])
                if fsky is not None:  # If fsky is defined, use the fsky value
                    res[i, j] = (
                     (C_AC + N_AC) * (C_BD + N_BD) + (C_AD + N_AD) * (C_BC + N_BC)
                     #(C_AC) * (C_BD) + (C_AD) * (C_BC)
                 ) /fsky
                else:  # Otherwise, use the fsky values from the input spectra
                     AC = keys[i, j, 0] +'x'+ keys[i, j, 2]
                     BD = keys[i, j, 1] +'x'+ keys[i, j, 3]
                     AD = keys[i, j, 0] +'x'+ keys[i, j, 3]
                     BC = keys[i, j, 1] +'x'+ keys[i, j, 2]
                     AB = keys[i, j, 0] +'x'+ keys[i, j, 1]
                     CD = keys[i, j, 2] +'x'+ keys[i, j, 3]
                     res[i, j] = (
                         np.sqrt(fskies[AC] * fskies[BD])
                         * (C_AC + N_AC)
                         * (C_BD + N_BD)
                         + np.sqrt(fskies[AD] * fskies[BC])
                         * (C_AD + N_AD)
                         * (C_BC + N_BC)
                     ) / (fskies[AB] * fskies[CD])
                res[j, i] = res[i, j]
        return res





def get_masked_sigma(
    n: int,
    absolute_lmin: int,
    absolute_lmax: int,
    gauss_keys: dict,
    sigma: np.ndarray,
    excluded_probes: list,
    lmins: dict = {},
    lmaxs: dict = {},
):
    """Mask the covariance matrix for the Gaussian case in certain ranges of multipoles.

    The covariance matrix is correctly built between lmin and lmax by the function "sigma". However, some observables might be missing in some multipole ranges, so we need to fill the matrix with zeros.

    Parameters:
        n (int):
            Number of fields.
        absolute_lmin (int):
            The minimum multipole to consider.
        absolute_lmax (int):
            The maximum multipole to consider.
        gauss_keys (dict):
            Keys for the covariance elements.
        sigma (ndarray):
            The covariance matrix.
        excluded_probes (list):
            List of probes to exclude.
        lmins (dict):
            The dictionary of minimum multipole to consider for each field pair.
        lmaxs (dict):
            The dictionary of maximum multipole to consider for each field pair.
    """
    n = int(n * (n + 1) / 2)
    mask = np.zeros(sigma.shape)

    for i in range(n):
        key = gauss_keys[i, i, 0] +'x'+ gauss_keys[i, i, 1]

        lmin = lmins.get(key, absolute_lmin)
        lmax = lmaxs.get(key, absolute_lmax)
        print(key, lmin, lmax)
        for ell in range(absolute_lmax + 1):
            if ell < lmin or ell > lmax:
                mask[i, :, ell] = 1
                mask[:, i, ell] = 1
            if excluded_probes is not None and key in excluded_probes:
                mask[i, :, ell] = 1
                mask[:, i, ell] = 1

    return np.ma.masked_array(sigma, mask)

def inv_sigma(lmin: int, lmax: int, masked_sigma):
    """Invert the covariance matrix of the Gaussian case.

    Inverts the previously calculated sigma ndarray. Note that some elements may be null, thus the covariance may be singular. If so, this also reduces the dimension of the matrix by deleting the corresponding row and column.

    Parameters:
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        masked_sigma (np.ma.masked_array):
            Previously computed and masked covariance matrix (not inverted).
    """
    res = []
    for ell in range(lmax + 1):
        # Here we need to remove the masked elements to get the non null covariance matrix
        new_dimension = np.count_nonzero(np.diag(masked_sigma.mask[:, :, ell]) == False)
        COV = masked_sigma[:, :, ell].compressed().reshape(new_dimension, new_dimension)
        # This check is not necessary in principle, but it is useful to avoid singular matrices
        if np.linalg.det(COV) == 0:
            idx = np.where(np.diag(COV) == 0)[0]
            COV = np.delete(COV, idx, axis=0)
            COV = np.delete(COV, idx, axis=1)

        res.append(np.linalg.inv(COV))
    return res[lmin:], masked_sigma.mask[:, :, lmin:]








def get_masked_derivates(n_fields, masked_sigma, derivative, ells):
    '''It applies the mask to the derivative of the power spectra. It is useful to avoid singular matrices.'''
    '''The masked_sigma is a masked array, so it is necessary to apply the mask to the derivative as well.'''
    n_probes = int(n_fields * (n_fields + 1) / 2)
    masked_derivative = []
    for ell in range(len(ells)):
        temp = []
        for i in range(int(n_probes)):
            if np.diag(masked_sigma.mask[:,:,ell+2])[i] == False:
                temp.append(derivative[ell,i,:])
        masked_derivative.append(temp)
    return masked_derivative
# make a function that takes the derivate and the sigma to get the fisher matrix




def get_fisher(ell,n_params,masked_derivative, inverse_sigma):
    '''This function computes the Fisher matrix from the masked derivative and the inverse of the covariance matrix.'''

    maksed_derivative_transposed = []
    fisher_matrix = []

    for j in range(len(ell)):
        maksed_derivative_transposed.append(np.transpose(masked_derivative[j]))

    for j in range(len(ell)):

        fisher_matrix.append(maksed_derivative_transposed[j]@inverse_sigma[0][j]@masked_derivative[j])
    
    # Initialize the sum with zeros
    sum_matrix = np.zeros((n_params, n_params))

    # Iterate over the matrices and accumulate the sum
    for matrix in fisher_matrix:
        sum_matrix += matrix
    
    inverse_fisher_matrix = np.linalg.inv(sum_matrix)
    #invert fisher to get covariance matrix of forecasts

    one_sigma_errors = np.sqrt(np.diagonal(inverse_fisher_matrix))
    #errors are squared on the diagonal

    return fisher_matrix, inverse_fisher_matrix, one_sigma_errors




def compute_fskies(n_fields, fields, fsky, use_min_fsky_for_crosses=False):
    """
    Computes the fskies dictionary based on input parameters and a choice to set cross fsky values to the minimum of the two fields' fsky values.

    Parameters:
    - n_fields: int, number of fields.
    - fields: list of str, names of the fields.
    - fsky: list of float, fsky values for each field.
    - use_min_fsky_for_crosses: bool, if True, sets fsky for crosses (TxW, ExW, PxW, etc.) to the minimum of the two fields' fsky values.

    Returns:
    - fskies: dict, keys are field combinations and values are their fsky values.
    """
    fskies = {}

    for i in range(n_fields):
        for j in range(i, n_fields):
            key = fields[i] + 'x' + fields[j]
            
            # Determine the fsky value based on the choice
            if use_min_fsky_for_crosses:
                # Set fsky to the minimum of the two fskys for the fields involved
                min_fsky_value = min(fsky[i], fsky[j])
                fskies[key] = min_fsky_value
            else:
                fskies[key] = np.sqrt(fsky[i] * fsky[j])

            # Ensure symmetry in fskies
            reverse_key = fields[j] + 'x' + fields[i]
            fskies[reverse_key] = fskies[key]

    return fskies





