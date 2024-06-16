import numpy as np
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




