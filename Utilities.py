'''File that contains utility functions for various tasks in the project.


Functions:

- get_keys(fields): Generate combinations of the given fields with 'x' as a delimiter.
- get_Gauss_keys_v3(fields): Generate keys for Gaussian likelihood based on the combinations of given fields.
- find_spectrum(input_dict, lmax, lmin, key): Find a spectrum in a given dictionary.
- txt2dict(txt, mapping=None, apply_ellfactor=None): Takes a txt file and convert it to a dictionary.
- print_errors(variables_dictionary, fid_values_dictionary, errors): Print the errors in a nice way.
- get_column_headers_with_index(file_path): Get the column headers with their index from a file.
- remove_symmetric_duplicates(obs_dict): Remove symmetric duplicates from the dictionary of observables.
- get_mapping(file_path, n_fields): Generates a mapping of column headers to their indices for cosmological data files.
- extract_fiducial_cl_from_CAMB(file_path_unlens, file_path_lens, column_mapping, n_counts, ls): Extracts and processes the fiducial cosmological power spectrum from CAMB output files.
- load_data(epsilon, var, col_num, base_path, directory_structure, lensed=False): Loads data from a specified file based on the given parameters.
- load_and_process_data(var, epsilon, obs, col_num, lensed, base_path, directory_structure): Loads and processes data for a given variable and perturbation.
- process_cl_data(vars, epsilon_values, column_mapping, ls, base_path, directory_structure): Processes power spectrum data for multiple cosmological variables across different perturbations in parallel.
- count_ws(fields): This function takes a list of strings 'fields' and returns the count of how many start with 'W'.
- filter_dictionary(data, exclude_chars, separator='x'): Filters a dictionary by removing entries based on specified exclusion criteria and ensuring uniqueness of symmetric keys.






'''








import numpy as np
import Derivate_coeff as dc
import concurrent.futures


from numpy import polyfit, polyval

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

import Fisher_Matrix as fm
import Derivatives as dv


# Ensure the rest of your functions follow here




def get_keys(fields):
    """Generate combinations of the given fields with 'x' as a delimiter."""
    n = len(fields)
    return [
        fields[i] + 'x' + fields[j]
        for i in range(n)
        for j in range(i, n)
    ]




def get_Gauss_keys_v3(fields):
    """Generate keys for Gaussian likelihood based on the combinations of given fields."""
    keys = get_keys(fields)
    n = len(keys)
    # Initialize a 3D array to store the keys
    res = np.zeros((n, n, 4), dtype=object)

    # Loop over all the elements in the covariance matrix
    for i in range(n):
        for j in range(n):
            # Split the keys and create a 4-element combination
            parts_i = keys[i].split('x')
            parts_j = keys[j].split('x')
            key = [parts_i[0], parts_i[1], parts_j[0], parts_j[1]]

            # Assign the key to the array
            res[i, j] = key

    # Return the keys
    return res





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

def txt2dict(txt, mapping=None, apply_ellfactor=None):
        """Takes a txt file and convert it to a dictionary. This requires a way to map the columns to the keys. Also, it is possible to apply an ell factor to the Cls.

        Parameters:
            txt (str):
                Path to txt file containing the spectra as columns.
            mapping (dict):
                Dictionary containing the mapping. Keywords will become the new keywords and values represent the index of the corresponding column.
        """
        # Define the ell values from the length of the txt file
        assert (
            mapping is not None
        ), "You must provide a way to map the columns of your txt to the keys of a dictionary"
        ls = np.arange(txt.shape[0], dtype=np.int64)
        res = {}
        # Loop over the mapping and extract the corresponding column from the txt file
        # and store it in the dictionary under the corresponding keyword
        for key, i in mapping.items():
            if apply_ellfactor:
                res[key] = txt[:, i] * ls * (ls + 1) / 2 / np.pi
            else:
                res[key] = np.insert(txt[:, i],[0,0],[0,0])
        return res






def print_errors(variables_dictionary, fid_values_dictionary, errors):
    '''This function prints the errors in a nice way'''
    print('DISCLAIMER: Omegab is ombh2 and Omegac is omch2')
    for i in range(len(variables_dictionary)):
        print('absolute sigma', variables_dictionary[i],'=',errors[i])
        print('relative sigma', variables_dictionary[i],'=',errors[i]/fid_values_dictionary[i])






def get_column_headers_with_index(file_path):
    '''Get the column headers with their index from a file.'''
    with open(file_path, 'r') as file:
        # Read the first line and split, but ignore the first '#' symbol
        headers = file.readline().strip().split()[1:]  # Skips the first '#'

    # Create a dictionary mapping headers to their column index
    header_to_index = {header: index for index, header in enumerate(headers)}

    return header_to_index

def remove_symmetric_duplicates(obs_dict):
    '''Remove symmetric duplicates from the dictionary of observables.'''
    keys_to_remove = []
    processed_pairs = set()

    for key in obs_dict:
        # Split the key and sort to find symmetric duplicates
        observables = key.split('x')
        if len(observables) != 2:
            continue  # Skip if not a pair

        sorted_pair = tuple(sorted(observables))
        if sorted_pair in processed_pairs:
            keys_to_remove.append(key)
        else:
            processed_pairs.add(sorted_pair)

    for key in keys_to_remove:
        del obs_dict[key]



def get_mapping(file_path, n_fields):
    """
    Generates a mapping of column headers to their indices for cosmological data files, 
    ensuring the number of probes matches the expected based on the number of fields.

    Args:
    file_path (str): Path to the data file from which to extract header information.
    n_fields (int): Number of fields (e.g., temperature, E-mode polarization) in the data.

    Returns:
    dict: A dictionary mapping column names to their respective indices.
    """
    # Calculate the expected number of probes from the number of fields
    n_probes = (n_fields * (n_fields + 1)) // 2
    
    # Extract column headers with their index positions
    column_mapping = get_column_headers_with_index(file_path)
    column_mapping.pop('L')
    remove_symmetric_duplicates(column_mapping)
    # Verify the correct number of probes based on the extracted headers
    if len(column_mapping.items()) != n_probes:
        print('The number of probes is not correct')
    
    # Remove entries for symmetric duplicates in the mapping

    
    # Remove the 'L' column as it is not considered a probe
    
    
    # Manually set specific indices for certain probes as required by file structure
    column_mapping['ExE'] = 2  # Forcing index for 'ExE' to 2 as specific to lensed files
    column_mapping['TxE'] = 4  # Forcing index for 'TxE' to 4 as specific to lensed files
    
    return column_mapping






def extract_fiducial_cl_from_CAMB(file_path_unlens, file_path_lens, column_mapping, n_counts, ls):
    """
    Extracts and processes the fiducial cosmological power spectrum from CAMB output files, 
    both unlensed and lensed, and applies normalization and scaling transformations.

    Args:
    file_path_unlens (str): Path to the unlensed data file.
    file_path_lens (str): Path to the lensed data file.
    column_mapping (dict): Mapping of observational types to column indices.
    n_counts (int): Number of additional power spectra to process beyond the CMB ones.
    ls (numpy.ndarray): Array of multipole values.

    Returns:
    dict: A dictionary containing processed power spectra indexed by type.
    """
    # Load data from the unlensed and lensed files using the provided column mappings
    clsdict = txt2dict(np.loadtxt(file_path_unlens), mapping=column_mapping, apply_ellfactor=None)
    clsdict2 = txt2dict(np.loadtxt(file_path_lens), mapping={'TxT': 1, 'ExE': 2, 'TxE': 4}, apply_ellfactor=None)

    # Update unlensed data with lensed equivalents for specific types
    for key in ['TxT', 'ExE', 'TxE']:
        clsdict[key] = clsdict2[key]

    # Adjust the ls array to include zeros at the beginning for indexing convenience
    ls2 = np.insert(ls, [0, 0], [0, 0])
    normfactor0 = (ls2 * (ls2 + 1)) / (2 * np.pi)  # Normalization factor based on multipoles

    # remove the scaling ell-factor and handle NaNs
    for key, value in clsdict.items():
        newvalue = value / normfactor0
        newvalue[np.isnan(newvalue)] = 0
        clsdict[key] = newvalue

    # Apply specific scaling factors to correlation terms involving CMB lensing (from deflection angle to convergence)
    for key in ['TxP', 'ExP', 'PxP']:
        if key in clsdict:
            p_count = key.count('P')
            scale = (0.5 * np.sqrt(ls * (ls + 1))) ** p_count
            clsdict[key][2:] *= scale

    # Apply scaling to other cross-correlation terms involving LSS probes
    for i in range(1, n_counts + 1):
        key = f'P' + 'x' + 'W' + str(i)
        if key in clsdict:
            clsdict[key][2:] *= 0.5 * np.sqrt(ls * (ls + 1))

    return clsdict





from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
from tqdm import tqdm

def load_data(epsilon, var, col_num, base_path, directory_structure, lensed=False):
    """
    Loads data from a specified file based on the given parameters.

    Args:
    epsilon (float): The perturbation value to specify the file.
    var (str): The variable name to specify the directory.
    col_num (int): The column number to extract from the file.
    base_path (str): The base path of the data files.
    directory_structure (str): The path pattern leading to the data files, including {var}.
    lensed (bool): A flag to determine if lensed or unlensed data should be loaded.

    Returns:
    numpy.ndarray: Data extracted from the specified column of the file.
    """
    # Determine the suffix for the file name based on the lensing status
    lensed_suffix = 'lensed' if lensed else 'unlensed'
    
    # Generate the file suffix based on the epsilon value
    if epsilon > 0:
        file_suffix = f"_pl_eps_{format(epsilon, '.1E').replace('.', 'p').replace('+', '')}.txt"
    elif epsilon < 0:
        file_suffix = f"_mn_eps_{format(abs(epsilon), '.1E').replace('.', 'p')}.txt"
    else:  # epsilon == 0
        file_suffix = "fiducial.txt"
    
    # Construct the full file path
    if epsilon == 0:
        full_directory = os.path.join(base_path,directory_structure.format(var='fiducial'),lensed_suffix, file_suffix)
    else:
        full_directory = os.path.join(base_path, directory_structure.format(var=var), lensed_suffix, file_suffix)
    
    # Load and return the specified column from the file
    data = np.loadtxt(full_directory, usecols=col_num)
    return data

def load_and_process_data(var, epsilon, obs, col_num, lensed, base_path, directory_structure):
    """
    Loads and processes data for a given variable and perturbation.

    Args:
    var (str): Variable name.
    epsilon (float): Perturbation value.
    obs (str): Observation type identifier.
    col_num (int): Column number to load from the file.
    lensed (bool): Specifies whether to load lensed or unlensed data.
    base_path (str): Base directory for data files.
    directory_structure (str): Directory structure leading to data files.

    Returns:
    tuple: A tuple containing variable name, observation type, epsilon, and the loaded data.
    """
    # Load data using the parameters provided
    data = load_data(epsilon, var, col_num, base_path, directory_structure, lensed=lensed)
    
    # Return all input parameters along with the loaded data for further processing
    return var, obs, epsilon, data




def process_cl_data(vars, epsilon_values, column_mapping, ls, base_path, directory_structure):
    """
    Processes power spectrum data for multiple cosmological variables across different perturbations in parallel.

    Args:
    vars (list of str): A list of variable names whose spectra need to be processed.
    epsilon_values (list of float): List of perturbation values applied to variables.
    column_mapping (dict): Mapping of observation types to column indices in the data files.
    ls (numpy array): Array of multipole values (l values) for which the data is processed.
    base_path (str): The base path where the data files are stored.
    directory_structure (str): The directory structure that leads to the data files, including placeholders.

    Returns:
    dict: A dictionary with keys as variable names and values as numpy arrays of processed data.
    """
    # Create an empty dictionary to hold processed data for each variable.
    cl_data = {var: np.zeros((len(ls), len(column_mapping), len(epsilon_values))) for var in vars}

    # Convert lists of epsilon values and column mapping keys to lists for easier indexing.
    epsilon_values_list = list(epsilon_values)
    column_keys_list = list(column_mapping.keys())

    # Create a list of tasks for parallel execution. Each task corresponds to loading and processing data for a specific combination of parameters.
    tasks = [(var, epsilon, obs, col_num, obs in ['TxT', 'ExE', 'TxE'], base_path, directory_structure)
             for var in vars
             for epsilon in epsilon_values
             for obs, col_num in column_mapping.items()]

    # Use a ProcessPoolExecutor to handle parallel execution of tasks.
    with ProcessPoolExecutor() as executor:
        # Map each future (task being executed) to its corresponding task for easier tracking.
        future_to_task = {executor.submit(load_and_process_data, *task): task for task in tasks}
        # Collect all futures to track their completion.
        futures = list(future_to_task.keys())
        
        # Iterate over each completed future and process the results.
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Data"):
            task = future_to_task[future]
            try:
                # When a future completes, extract the result and update the cl_data structure.
                var, obs, epsilon, data = future.result()
                epsilon_index = epsilon_values_list.index(epsilon)
                obs_index = column_keys_list.index(obs)
                cl_data[var][:, obs_index, epsilon_index] = data[:]
            except Exception as exc:
                # If an error occurs during the execution of a task, print the error and the task details.
                print(f'{task} generated an exception: {exc}')
    # Return the processed data.
    return cl_data



def count_ws(fields):
    # This function takes a list of strings 'fields' and returns the count of how many start with 'W'

    # We use a generator expression to iterate through each field in the list
    # 'field.startswith('W')' evaluates to True if the field name starts with 'W'
    # sum(1 for ...) sums up 1 for each field that starts with 'W', effectively counting them
    return sum(1 for field in fields if field.startswith('W'))






def filter_dictionary(data, exclude_chars, separator='x'):
    """
    Filters a dictionary by removing entries based on specified exclusion criteria and ensuring uniqueness of symmetric keys.

    Parameters:
    - data (dict): The dictionary to be filtered.
    - exclude_chars (list of str): Characters that if found in a key, will lead to its exclusion from the final dictionary.
    - separator (str, optional): A character used to split the keys into components to identify symmetric entries. Default is 'x'.

    The function operates in two main steps:
    1. Exclusion of keys: Removes any key-value pairs where the key contains any of the characters specified in `exclude_chars`.
    2. Symmetry reduction: Further filters the dictionary by removing symmetric entries based on the separator. For instance, for a key "AxB", a subsequent key "BxA" would be excluded if already considered. This is done by sorting the components of each key and ensuring each unique combination appears only once.

    Returns:
    - dict: A filtered dictionary with keys that do not contain specified characters and do not have symmetric duplicates.

    Example usage:
    >>> data = {'AxB': 1, 'BxA': 2, 'CxD': 3, 'DxC': 4, 'ExF': 5}
    >>> filter_dictionary(data, ['E'], 'x')
    {'AxB': 1, 'CxD': 3}
    """
    # Create a dictionary excluding keys that contain any character from `exclude_chars`
    intermediate_dict = {key: value for key, value in data.items() if not any(char in key for char in exclude_chars)}
    
    # Prepare to collect unique combinations of key components
    final_dict = {}
    seen_pairs = set()
    
    for key in intermediate_dict:
        # Split the key into components by the separator
        components = key.split(separator)
        
        # Sort the components to standardize their order, making them comparable
        sorted_components = tuple(sorted(components))
        
        # If this sorted tuple hasn't been added yet, proceed to add it
        if sorted_components not in seen_pairs:
            # Mark this combination as seen
            seen_pairs.add(sorted_components)
            # Add the original key and its value to the final dictionary
            final_dict[key] = intermediate_dict[key]

    # Return the dictionary filtered from duplicates and excluded characters
    return final_dict










