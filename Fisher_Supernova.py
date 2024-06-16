import sys, platform, os

camb_installation_path = '/home/guglielmo/Desktop/eftcamb_new_1'  
camb_path = os.path.realpath(os.path.join(os.getcwd(),camb_installation_path))
sys.path.insert(0,camb_path)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import camb
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
from scipy.special import erf
import pandas as pd
import shutil
import tempfile
from scipy.integrate import simps
try:
    from tqdm.notebook import tqdm  # This is specifically for Jupyter notebooks
except ImportError:
    from tqdm import tqdm  # Standard console version as a fallback
import time


class FisherSN:
    """
    Class to hold parameters for generating mock supernova (SN) data and computing the Fisher matrix.
    
    Attributes:
    - number_SN_windows (int): Number of SN survey bins in redshift.
    - SN_number (np.array): Array containing the number of SN in each redshift bin.
    - SN_redshift_start (np.array): Array containing the starting redshift for each bin.
    - SN_redshift_end (np.array): Array containing the ending redshift for each bin.
    - total_SN_number (int): Total number of supernovae.
    - SN_Fisher_MC_samples (int): Number of realizations over which the SN Fisher is computed.
    - alpha_SN (float): Fiducial value of the SN alpha parameter.
    - beta_SN (float): Fiducial value of the SN beta parameter.
    - M0_SN (float): Fiducial value of the SN M0 (absolute magnitude).
    - color_dispersion (float): Dispersion of the color of the mock SN catalog.
    - stretch_dispersion (float): Dispersion of the stretch of the mock SN catalog.
    - magnitude_sigma (float): Error on the absolute magnitude.
    - c_sigmaz (float): Error in the redshift determination.
    - sigma_lens_0 (float): Lensing error coefficient.
    - dcolor_offset (float): Error in the color at redshift zero.
    - dcolor_zcorr (float): Coefficient for the redshift dependence of the color error.
    - dshape_offset (float): Error in the stretch at redshift zero.
    - dshape_zcorr (float): Coefficient for the redshift dependence of the stretch error.
    - cov_ms_offset (float): Covariance between the error in magnitude and stretch at redshift zero.
    - cov_ms_zcorr (float): Coefficient for the redshift dependence of the covariance between the error in magnitude and stretch.
    - cov_mc_offset (float): Covariance between the error in magnitude and color at redshift zero.
    - cov_mc_zcorr (float): Coefficient for the redshift dependence of the covariance between the error in magnitude and color.
    - cov_sc_offset (float): Covariance between the error in stretch and color at redshift zero.
    - cov_sc_zcorr (float): Coefficient for the redshift dependence of the covariance between the error in stretch and color.
    """
    def __init__(self, number_SN_windows, SN_number, SN_redshift_start, SN_redshift_end, 
                 total_SN_number, SN_Fisher_MC_samples, alpha_SN, beta_SN, M0_SN, 
                 color_dispersion, stretch_dispersion, magnitude_sigma, c_sigmaz, 
                 sigma_lens_0, dcolor_offset, dcolor_zcorr, dshape_offset, dshape_zcorr, 
                 cov_ms_offset, cov_ms_zcorr, cov_mc_offset, cov_mc_zcorr, 
                 cov_sc_offset, cov_sc_zcorr):
        self.number_SN_windows = number_SN_windows
        self.SN_number = np.array(SN_number, dtype=int)
        self.SN_redshift_start = np.array(SN_redshift_start)
        self.SN_redshift_end = np.array(SN_redshift_end)
        self.total_SN_number = total_SN_number
        self.SN_Fisher_MC_samples = SN_Fisher_MC_samples
        self.alpha_SN = alpha_SN
        self.beta_SN = beta_SN
        self.M0_SN = M0_SN
        self.color_dispersion = color_dispersion
        self.stretch_dispersion = stretch_dispersion
        self.magnitude_sigma = magnitude_sigma
        self.c_sigmaz = c_sigmaz
        self.sigma_lens_0 = sigma_lens_0
        self.dcolor_offset = dcolor_offset
        self.dcolor_zcorr = dcolor_zcorr
        self.dshape_offset = dshape_offset
        self.dshape_zcorr = dshape_zcorr
        self.cov_ms_offset = cov_ms_offset
        self.cov_ms_zcorr = cov_ms_zcorr
        self.cov_mc_offset = cov_mc_offset
        self.cov_mc_zcorr = cov_mc_zcorr
        self.cov_sc_offset = cov_sc_offset
        self.cov_sc_zcorr = cov_sc_zcorr






class CAMBparameters:
    """
    Class to hold and access CAMB parameters, allowing for variations for derivatives calculation.
    """
    def __init__(self, As, ns, H0, ombh2, omch2, tau):
        self.As = As
        self.ns = ns
        self.H0 = H0
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.tau = tau
        self.omh2 = ombh2+omch2  # Including omh2 if needed for specific calculations

    def vary_H0(self, j):
        """
        Vary H0 and related parameters.
        """
        H0_var = self.H0 * (1 + j)
        ombh2_var = self.ombh2 * (1 + j)**2
        omch2_var = self.omch2 * (1 + j)**2
        return {'H0': H0_var, 'ombh2': ombh2_var, 'omch2': omch2_var, 'As': self.As, 'ns': self.ns, 'tau': self.tau}

    def vary_omch2(self, j):
        """
        Vary omch2 according to a specific relationship involving omh2.
        """
        omch2_var = self.omch2 * (1 + j * self.omh2 / self.omch2)
        return {'H0': self.H0, 'ombh2': self.ombh2, 'omch2': omch2_var, 'As': self.As, 'ns': self.ns, 'tau': self.tau}






def calculate_bin_width(z_min, z_max, number_of_bins):
    return (z_max - z_min) / number_of_bins



def calculate_number_of_bins(z_min, z_max, desired_bin_width):
    return int(np.ceil((z_max - z_min) / desired_bin_width))


def init_random():
    np.random.seed()

def random_uniform(start, end):
    return np.random.uniform(start, end)

def random_gaussian(mean, sigma):
    return np.random.normal(mean, sigma)

def generate_sn_mock_data(P, FP):
    """
    Generates a mock dataset of supernovae with redshift, color, and stretch values based on the given parameters.

    Parameters:
    - P: CAMBparams object, structured type containing simulation parameters.
    - FP: Instance of CosmicFishFisherSN, structured type with attributes detailing the supernova simulation setup.
    
    Returns:
    - sn_mock: A 2D numpy array with dimensions 3 x total_SN_number containing the mock supernova data.
    """
    init_random()
    sn_mock = np.zeros((3, FP.total_SN_number))
    index = 0
    for j in range(FP.number_SN_windows):
        for k in range(FP.SN_number[j]):
            sn_mock[0, index] = random_uniform(FP.SN_redshift_start[j], FP.SN_redshift_end[j])
            sn_mock[1, index] = random_gaussian(0, FP.color_dispersion)
            sn_mock[2, index] = random_gaussian(0, FP.stretch_dispersion)
            index += 1
    return sn_mock


def calculate_luminosity_distance(As_func,ns_func,H0_func,ombh2_func,omch2_func,tau_func,redshifts):
    # Define the cosmological parameters
    eftcamb_GR = {'EFTflag': 0}

    # Initialize CAMB parameters
    pars_GR = camb.set_params(lmax=1500,
                              As=As_func,
                              ns=ns_func,
                              H0=H0_func,
                              ombh2=ombh2_func,
                              omch2=omch2_func,
                              tau=tau_func,
                              WantCls=True,
                              WantTransfer=True,
                              WantScalars=True,
                              WantTensors=False,
                              WantVectors=False,
                              WantDerivedParameters=True,
                              NonLinear=0,
                              **eftcamb_GR)

    # Set accuracy
    pars_GR.set_accuracy(AccuracyBoost=2, lAccuracyBoost=2, lSampleBoost=2)

    # Set parameters for lmax
    pars_GR.set_for_lmax(lmax=1500, lens_potential_accuracy=0)

    # Source terms adjustments
    pars_GR.SourceTerms.limber_windows = True
    pars_GR.SourceTerms.limber_phi_lmin = 100
    pars_GR.SourceTerms.counts_density = True
    pars_GR.SourceTerms.counts_redshift = False
    pars_GR.SourceTerms.counts_lensing = False
    pars_GR.SourceTerms.counts_velocity = False
    pars_GR.SourceTerms.counts_radial = False
    pars_GR.SourceTerms.counts_timedelay = False
    pars_GR.SourceTerms.counts_ISW = False
    pars_GR.SourceTerms.counts_potential = False
    pars_GR.SourceTerms.counts_evolve = False

    # Get results
    results_GR = camb.get_results(pars_GR)

    # Calculate luminosity distance for each z in the array
    distance = [results_GR.luminosity_distance(value) for value in redshifts]

    return distance




def generate_sn_mock_covariance(P, FP, redshift):
    """
    Generates a mock covariance matrix for the supernova data.

    Parameters:
    - P: CAMBparams object, structured type containing simulation parameters.
    - FP: Instance of CosmicFishFisherSN, containing parameters for supernova calculations.
    - sn_mock: A 2D numpy array with mock supernova data.

    Returns:
    - sn_mock_covariance: A numpy array containing the covariance data for each supernova.
    """
    total_sn_number = FP.total_SN_number
    sn_mock_covariance = np.zeros(total_sn_number)
    c = 299792.458  # Speed of light in km/s (for conversion purposes)
    
    for i in range(total_sn_number):
    
        lensing_term = (FP.sigma_lens_0 * redshift[i]) ** 2
        redshift_term = (5 / redshift[i] / np.log(10) * FP.c_sigmaz / c * 1000) ** 2
        magnitude_term = (FP.magnitude_sigma) ** 2

        systematic_effects = FP.beta_SN**2 * (FP.dcolor_offset + FP.dcolor_zcorr * redshift[i]**2)**2 \
                             + FP.alpha_SN**2 * (FP.dshape_offset + FP.dshape_zcorr * redshift[i]**2)**2 \
                             + 2 * FP.alpha_SN * (FP.cov_ms_offset + FP.cov_ms_zcorr * redshift[i]**2) \
                             - 2 * FP.beta_SN * (FP.cov_mc_offset + FP.cov_mc_zcorr * redshift[i]**2) \
                             - 2 * FP.alpha_SN * FP.beta_SN * (FP.cov_sc_offset + FP.cov_sc_zcorr * redshift[i]**2)

        sn_mock_covariance[i] = magnitude_term + lensing_term + redshift_term + systematic_effects

    return sn_mock_covariance




def observed_sn_magnitude(P, FP, redshift, color, stretch, num):
    """
    Calculates the observed magnitude of a set of supernovae based on the input cosmological parameters.

    Parameters:
    - P: Dictionary of cosmological parameters.
    - FP: Instance of CosmicFishFisherSN, containing parameters for supernova calculations.
    - redshift: Numpy array with the values of redshift.
    - color: Numpy array with the values of color.
    - stretch: Numpy array with the values of stretch.
    - num: Number of supernovae.

    Returns:
    - sn_magnitude: Numpy array containing the calculated supernova magnitudes.
    - err: Error code indicating the status of the calculation.
    """
    sn_magnitude = np.zeros(num)
    err = 0

    try:
        # Call a function to compute luminosity distance for all redshifts
        # Ensure that this function can handle dictionary parameters
        luminosity_distances = calculate_luminosity_distance(P['As'], P['ns'], P['H0'], P['ombh2'], P['omch2'], P['tau'], redshift)
        log_distances = np.log10(luminosity_distances)

        for i in range(num):
            sn_magnitude[i] = 5.0 * log_distances[i] - FP.alpha_SN * stretch[i] + FP.beta_SN * color[i] + FP.M0_SN + 25.0

    except Exception as e:
        print(f"An error occurred: {e}")
        err = 4  # Example error code for unspecified errors

    return sn_magnitude, err


def calculate_derivative(camb_params, FP, redshift, color, stretch, num, param_to_vary, epsilon):
    """
    Calculate the derivative of supernova magnitude with respect to a specified parameter using the central difference method.
    
    Parameters:
    - camb_params: Instance of CAMBparameters, containing cosmological parameters.
    - FP: Instance of CosmicFishFisherSN, containing supernova parameters.
    - redshift: Array of redshift values.
    - color: Array of color values.
    - stretch: Array of stretch values.
    - num: Number of supernovae.
    - param_to_vary: String indicating which parameter to vary ('H0', 'omch2', etc.).
    - epsilon: Small fractional change to apply to the parameter.
    
    Returns:
    - derivative_magnitudes: Array containing the derivative of magnitudes.
    - err: Error code indicating the success or failure of the calculation.
    """
    # Initialize arrays for plus and minus variations
    magnitudes_plus = np.zeros(num)
    magnitudes_minus = np.zeros(num)
    
    # Parameter variation for both positive and negative epsilon
    for sign, magnitudes in [(-epsilon, magnitudes_minus), (epsilon, magnitudes_plus)]:
        # Get varied parameters
        if param_to_vary == 'H0':
            varied_params = camb_params.vary_H0(sign)
        elif param_to_vary == 'omch2':
            varied_params = camb_params.vary_omch2(sign)
        else:
            print(f"Parameter {param_to_vary} variation is not supported.")
            return None, 1  # Error code 1 for unsupported parameter


        # Calculate supernova magnitude for varied parameters
        sn_magnitude, err = observed_sn_magnitude(varied_params, FP, redshift, color, stretch, num)
        if err != 0:
            print("Error in magnitude calculation with varied parameters.")
            return None, err
        
        magnitudes[:] = sn_magnitude  # Store computed magnitudes

    # Compute the central difference
    derivative_magnitudes = (magnitudes_plus - magnitudes_minus) / (2 * epsilon * getattr(camb_params, param_to_vary))

    return derivative_magnitudes, 0  # Return derivatives and error code 0 for success
























def fisher_sn(P, FP, params, outroot=None):
    """
    Computes the SN Fisher matrix using Monte Carlo simulations.
    
    Parameters:
    - P: CAMBparams object, structured type containing simulation parameters.
    - FP: Instance of CosmicFishFisherSN, containing parameters for supernova calculations.
    - params: parameters in the Fisher matrix.
    - outroot: Optional filename for dumping outputs if feedback is greater than 1.
    
    Returns:
    - Fisher_Matrix: Computed Fisher matrix as a NumPy array.
    """
    # Allocate arrays for the simulation data
    num_param = len(params)
    sn_mock = np.zeros((3, FP.total_SN_number))
    sn_mock_covariance = np.zeros(FP.total_SN_number)

    sn_derivative = np.zeros(FP.total_SN_number)

    sn_derivative_array = np.zeros((num_param, FP.total_SN_number))

    MC_Fisher_Matrix = np.zeros((num_param, num_param))

    # Perform Monte Carlo simulations
    time_start_MC = time.time()
    for MC_i in tqdm(range(1, FP.SN_Fisher_MC_samples + 1)):
        sn_mock = generate_sn_mock_data(P, FP)
        sn_mock_covariance= generate_sn_mock_covariance(P, FP, sn_mock[0])


        # Compute derivatives for each parameter
        for ind,variable in enumerate(params):
            sn_derivative,err=calculate_derivative(P, FP, sn_mock[0], sn_mock[1], sn_mock[2], len(sn_mock[0]), variable, 0.01)
            sn_derivative_array[ind, :] = sn_derivative


        # Compute Fisher matrix elements
        for ind in range(num_param):
            for ind2 in range(num_param):
                
                temp2 = np.sum(sn_derivative_array[ind, :] * sn_derivative_array[ind2, :] / sn_mock_covariance)

  
                MC_Fisher_Matrix[ind, ind2] += temp2

    # Average the Monte Carlo results
    Fisher_Matrix = MC_Fisher_Matrix/FP.SN_Fisher_MC_samples
    time_end_MC = time.time()

    if outroot and FP.cosmicfish_feedback >= 1:
        print(f"Total time for MC Fisher matrix computation: {time_end_MC - time_start_MC} s")

    return Fisher_Matrix

# Functions generate_sn_mock_data, generate_sn_mock_covariance, observed_sn_magnitude,
# observed_sn_magnitude_derivative, and save_sn_mock_to_file need to be defined or properly adapted.
