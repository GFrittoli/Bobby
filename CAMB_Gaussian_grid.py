'''
File containing functions to grid Gaussian distributions. This file is an example valid for General relativity.


Functions:
- dndz: Calculate the redshift distribution function dN/dz.
- b: Calculate the bias factor.
- dndz_bin: Calculate the redshift distribution for a given bin.
- calculate_dndz: Calculate the redshift distribution and bias variations for all bins.
- spectra_dens: Compute the power spectra for a given set of parameters and bias.
- save_spectra: Save the computed power spectra to a file.
- generate_spectra_for_parameter: Generate power spectra for a given parameter variation.
- generate_spectra_for_bias: Generate power spectra for bias variations.
- compute_all_spectra: Compute power spectra for all parameters and biases, producing the fiducial spectra only once.






'''







import sys, platform, os

camb_installation_path = '/home/guglielmo/Desktop/eftcamb_new_1' 
camb_path = os.path.realpath(camb_installation_path)
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
from scipy.special import erf
import pandas as pd
import shutil
import tempfile

import numpy as np
from scipy.special import erf

# Define the CAMBparameters class
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
        self.omh2 = ombh2 + omch2  # Including omh2 if needed for specific calculations


class GalaxySurveyInfo:
    def __init__(self, zz, z_med, z0, c_b, z_b, sigma_b, c_0, z_0, sigma_0, f_out, z_bin_m, z_bin_p, n_bins):
        self.zz = zz
        self.z_med = z_med
        self.z0 = z0
        self.c_b = c_b
        self.z_b = z_b
        self.sigma_b = sigma_b
        self.c_0 = c_0
        self.z_0 = z_0
        self.sigma_0 = sigma_0
        self.f_out = f_out
        self.z_bin_m = z_bin_m
        self.z_bin_p = z_bin_p
        self.z_mid = [(z_bin_p[i + 1] + z_bin_m[i]) / 2 for i in range(len(z_bin_m) - 1)]
        self.n_bins = n_bins


def dndz(z, zm):
    """
    Calculate the redshift distribution function dN/dz.

    Parameters:
    - z: Redshift value or array of redshift values.
    - zm: Redshift scale factor.

    Returns:
    - dN/dz value(s).
    """
    return ((z / zm) ** 2) * np.exp(-(z / zm) ** (3. / 2.))

def b(z):
    """
    Calculate the bias factor.

    Parameters:
    - z: Redshift value.

    Returns:
    - Bias factor.
    """
    return np.sqrt(1 + z)

def dndz_bin(z, zm, zmin, zmax, c_b, z_b, sigma_b, c_0, z_0, sigma_0, f_out):
    #for Guassian case
    return f_out * np.exp(-0.5 * ((z - z_b) / sigma_b) ** 2) * dndz(z, zm)
def calculate_dndz(galaxy_info, epsilon):
    """
    Calculate the redshift distribution and bias variations for all bins.

    Parameters:
    - galaxy_info: Instance of GalaxySurveyInfo containing survey parameters.
    - epsilon: Array of perturbation values.

    Returns:
    - dndz_bins: List of redshift distributions for each bin.
    - bz_step_fid: Bias step function for the fiducial model.
    - variations_lists: List of bias variations for each bin.
    """
    z0 = galaxy_info.z_med / np.sqrt(2)  # Calculate z0
    dndz_tot = dndz(galaxy_info.zz, z0)  # Calculate total dndz

    # Calculate dndz for each bin
    dndz_bins = [
        dndz_bin(galaxy_info.zz, galaxy_info.z_med, galaxy_info.z_bin_m[i], galaxy_info.z_bin_p[i + 1], 
                 galaxy_info.c_b, galaxy_info.z_b, galaxy_info.sigma_b, galaxy_info.c_0, galaxy_info.z_0, 
                 galaxy_info.sigma_0, galaxy_info.f_out) 
        for i in range(len(galaxy_info.z_bin_m) - 1)
    ]

    # Create a list of indices for each bin
    kk = [[] for _ in range(len(galaxy_info.z_mid))]
    for i in range(len(galaxy_info.zz)):
        for j in range(len(galaxy_info.z_mid)):
            kk[j] = np.where((galaxy_info.zz >= galaxy_info.z_bin_m[j]) & (galaxy_info.zz <= galaxy_info.z_bin_p[j + 1]))

    # Calculate the bias step function for the fiducial model
    bz_step_fid = np.zeros(len(galaxy_info.zz))
    for i in range(len(galaxy_info.z_mid)):
        bz_step_fid[kk[i]] = b(galaxy_info.z_mid[i])

    # Initialize lists for variations in each of the 10 bins
    fid = int((len(epsilon) + 1) / 2) - 1
    variations_lists = [[] for _ in range(galaxy_info.n_bins)]  # One list for each bin

    # Iterate over each bin to create its variations
    for bin_index in range(galaxy_info.n_bins):  # Assuming 10 bins as per the initial setup
        for eps in epsilon:
            # Make a copy of the original bz_step for each variation
            bz_variation = np.copy(bz_step_fid)
            
            # Apply epsilon adjustment to the current bin only
            bz_variation[kk[bin_index]] = b(galaxy_info.z_mid[bin_index]) * (1 + eps)
            
            # Append this variation to the corresponding list
            variations_lists[bin_index].append(bz_variation)
    
    return dndz_bins, bz_step_fid, variations_lists


def spectra_dens(params, bz_step, dndz_bins, zz,n_bins, lmax=1500):
    eftcamb_GR = {'EFTflag': 0}                                                                     #COMMENT THIS LINE IF YOU WANT TO USE CAMB AND DONT HAVE EFTCAMB
    pars_GR = camb.set_params(lmax=lmax,
                              As=params.As,
                              ns=params.ns,
                              H0=params.H0,
                              ombh2=params.ombh2,
                              omch2=params.omch2,
                              tau=params.tau,
                              WantCls=True,
                              WantTransfer=True,
                              WantScalars=True,
                              NonLinear='NonLinear_both',
                              halofit_version='takahashi',
                              **eftcamb_GR)                                                        #COMMENT THIS LINE IF YOU WANT TO USE CAMB AND DONT HAVE EFTCAMB

    pars_GR.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars_GR.Accuracy.AccuracyBoost = 1.6
    pars_GR.SourceWindows = [SplinedSourceWindow(source_type='counts', bias_z=bz_step, z=zz, W=dndz_bins[i]) for i in range(n_bins)]
    pars_GR.SourceTerms.limber_windows = True
    pars_GR.SourceTerms.limber_phi_lmin = 200
    pars_GR.SourceTerms.counts_density = True
    pars_GR.SourceTerms.counts_redshift = False
    pars_GR.SourceTerms.counts = False
    pars_GR.SourceTerms.counts_velocity = False
    pars_GR.SourceTerms.counts_radial = False
    pars_GR.SourceTerms.counts_timedelay = False
    pars_GR.SourceTerms.counts_ISW = False
    pars_GR.SourceTerms.counts_potential = False
    pars_GR.SourceTerms.counts_evolve = False
    pars_GR.NonLinearModel.set_params(halofit_version='takahashi')
    results_GR = camb.get_results(pars_GR)

    cls_unlens = results_GR.get_cmb_unlensed_scalar_array_dict(lmax=lmax, CMB_unit=None, raw_cl=False)
    cls_lens = results_GR.get_lensed_scalar_cls(lmax=lmax, CMB_unit=None, raw_cl=False)
    cls_lens_dict = {'TxT': cls_lens[:, 0], 'ExE': cls_lens[:, 1], 'BxB': cls_lens[:, 2], 'TxE': cls_lens[:, 3]}

    return [cls_unlens, cls_lens_dict]

def find_s8(params, lmax=1500):
    eftcamb_GR = {'EFTflag': 0}
    pars_GR = camb.set_params(lmax=lmax,
                              As=params.As,
                              ns=params.ns,
                              H0=params.H0,
                              ombh2=params.ombh2,
                              omch2=params.omch2,
                              tau=params.tau,
                              WantCls=True,
                              WantTransfer=True,
                              WantScalars=True,
                              NonLinear='NonLinear_both',
                              **eftcamb_GR)

    pars_GR.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars_GR.Accuracy.AccuracyBoost = 1.6
    results_GR = camb.get_results(pars_GR)
    return np.array(results_GR.get_sigma8())

def save_spectra(cls_unlens, cls_lens, output_dir, output_dir1, j, lmax=1500):
    ls = np.arange(2, lmax + 1)
    values1 = np.zeros((lmax - 1, len(cls_unlens) + 1))
    values1[:, 0] = ls
    values2 = np.zeros((lmax - 1, len(cls_lens) + 1))
    values2[:, 0] = ls
    HEAD_unlens = 'L' + 5 * ' '
    HEAD_lens = 'L' + 5 * ' '

    for ii, (name, data) in enumerate(cls_unlens.items(), start=1):
        values1[:, ii] = data[2:lmax + 1]
        HEAD_unlens += f'{name:^10}' + 4 * ' '

    for ii, (name, data) in enumerate(cls_lens.items(), start=1):
        values2[:, ii] = data[2:lmax + 1]
        HEAD_lens += f'{name:^10}' + 4 * ' '

    if j == 0:
        filename_suffix = 'fiducial.txt'
    else:
        string_j = '{:1.1E}'.format(abs(j)).replace('.', 'p')
        filename_suffix = f'_pl_eps_{string_j}.txt' if j > 0 else f'_mn_eps_{string_j}.txt'

    np.savetxt(os.path.join(output_dir, filename_suffix), values1, fmt=' '.join(['%-4d'] + ['% -10.13e'] * len(cls_unlens)),
               delimiter=3 * ' ', header=HEAD_unlens)
    np.savetxt(os.path.join(output_dir1, filename_suffix), values2, fmt=' '.join(['%-4d'] + ['% -10.13e'] * len(cls_lens)),
               delimiter=3 * ' ', header=HEAD_lens)



def generate_spectra_for_parameter(param_name, fiducial_params, bz_step_fid, dndz_bins,zz,n_bins, epsilon, path_save, compute_fiducial=True):
    # Precompute s8_ref
    s8_ref = find_s8(fiducial_params)

    # Define the parameter perturbation functions
    params_funcs = {
        "h0": lambda j: CAMBparameters(fiducial_params.As, fiducial_params.ns, fiducial_params.H0 * (1 + j), fiducial_params.ombh2 * (1 + j) ** 2, fiducial_params.omch2 * (1 + j) ** 2, fiducial_params.tau),
        "Omegab": lambda j: CAMBparameters(fiducial_params.As, fiducial_params.ns, fiducial_params.H0, fiducial_params.ombh2 * (1 + j), fiducial_params.omch2 - fiducial_params.ombh2 * j, fiducial_params.tau),
        "Omegam": lambda j: CAMBparameters(fiducial_params.As, fiducial_params.ns, fiducial_params.H0, fiducial_params.ombh2, fiducial_params.omch2 * (1 + j * fiducial_params.omh2 / fiducial_params.omch2), fiducial_params.tau),
        "ns": lambda j: CAMBparameters(fiducial_params.As, fiducial_params.ns * (1 + j), fiducial_params.H0, fiducial_params.ombh2, fiducial_params.omch2, fiducial_params.tau),
        "tau": lambda j: CAMBparameters(fiducial_params.As, fiducial_params.ns, fiducial_params.H0, fiducial_params.ombh2, fiducial_params.omch2, fiducial_params.tau * (1 + j)),
        "ass": lambda j: CAMBparameters(np.exp(np.log(fiducial_params.As * 1e10) * (1 + j)) / (10 ** 10), fiducial_params.ns, fiducial_params.H0, fiducial_params.ombh2, fiducial_params.omch2, fiducial_params.tau)
    }

    if param_name not in params_funcs:
        raise ValueError(f"Unknown parameter: {param_name}")

    # Prepare directories and compute spectra
    output_dir = os.path.join(path_save, param_name, 'unlensed')
    output_dir1 = os.path.join(path_save, param_name, 'lensed')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir1, exist_ok=True)

    for j in epsilon:
        if j == 0 and not compute_fiducial:
            continue
        
        perturbed_params = params_funcs[param_name](j)


        cls_ulens, cls_lens = spectra_dens(perturbed_params, bz_step_fid, dndz_bins, zz,n_bins)
        save_spectra(cls_ulens, cls_lens, output_dir, output_dir1, j)

def generate_spectra_for_bias(fiducial_params, bz_step_fid, variations_lists, dndz_bins, zz,n_bins, epsilon, path_save, compute_fiducial=True):
    for kkk in range(n_bins):
        output_dir = os.path.join(path_save, 'b' + str(kkk + 1), 'unlensed')
        output_dir1 = os.path.join(path_save, 'b' + str(kkk + 1), 'lensed')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir1, exist_ok=True)

        for index, j in enumerate(epsilon):
            if j == 0 and not compute_fiducial:
                continue
            
            cls_ulens, cls_lens = spectra_dens(fiducial_params, variations_lists[kkk][index], dndz_bins, zz,n_bins)
            save_spectra(cls_ulens, cls_lens, output_dir, output_dir1, j)



def compute_all_spectra(listed_params, fiducial_params, galaxy_info, epsilon, path_save, compute_fiducial=True, compute_only="all"):
    """
    Compute spectra for all parameters and biases, producing the fiducial spectra only once.

    Parameters:
    - listed_params: List of parameters to perturb.
    - fiducial_params: Instance of CAMBparameters containing the fiducial values.
    - galaxy_info: Instance of GalaxySurveyInfo containing survey parameters.
    - epsilon: Array of perturbation values.
    - path_save: Path where the output should be saved.
    - compute_fiducial: Boolean flag to determine whether to compute fiducial spectra for epsilon = 0.
    - compute_only: Specify "all", "params", or "bias" to control which computations to perform.
    """
  
    # Prepare bias steps and bins
    dndz_bins, bz_step_fid, variations_lists = calculate_dndz(galaxy_info, epsilon)

    # Compute fiducial spectra if required
    if compute_fiducial:
        cls_ulens, cls_lens = spectra_dens(fiducial_params, bz_step_fid, dndz_bins, galaxy_info.zz,galaxy_info.n_bins)
        
        # Save fiducial spectra
        output_dir = os.path.join(path_save, 'fiducial', 'unlensed')
        output_dir1 = os.path.join(path_save, 'fiducial', 'lensed')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir1, exist_ok=True)
        save_spectra(cls_ulens, cls_lens, output_dir, output_dir1, 0)

    # Compute spectra for each parameter variation if specified
    if compute_only in ["all", "params"]:
        for param_name in listed_params:
            generate_spectra_for_parameter(param_name, fiducial_params, bz_step_fid, dndz_bins, galaxy_info.zz,galaxy_info.n_bins, epsilon, path_save, compute_fiducial=False)

    # Compute spectra for bias variations if specified
    if compute_only in ["all", "bias"]:
        generate_spectra_for_bias(fiducial_params, bz_step_fid, variations_lists, dndz_bins, galaxy_info.zz,galaxy_info.n_bins, epsilon, path_save, compute_fiducial=False)
