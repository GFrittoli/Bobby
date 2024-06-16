'''File containing functions to convert CLASS files to CAMB files and vice versa.



IN PROGRESS: This file is currently in progress and may not be fully functional or complete. Please check back later for updates.




'''










def map_keys(df_keys):
    """
    Maps a list of DataFrame column keys to a new format based on specified patterns in the keys. This specifically works to convert CLASS files to CAMB files.

    Parameters:
    - df_keys (list of str): A list of keys from a DataFrame, which typically include indices enclosed in brackets.

    This function transforms each key by extracting numerical indices from within brackets and formatting them into a new string pattern. The transformation assumes that the keys are formatted with indices in brackets like "X[1][2]". It constructs a new key in the format of 'W' followed by the first index, then 'xW', and then the second index.

    Returns:
    - dict: A dictionary where each original key from `df_keys` is mapped to its new formatted key.

    Example usage:
    >>> df_keys = ['X[1][2]', 'Y[3][4]']
    >>> map_keys(df_keys)
    {'X[1][2]': 'W1xW2', 'Y[3][4]': 'W3xW4'}
    """
    # Define the logic to convert DataFrame columns to a new key format
    return {
        df_key: 'W' + df_key.split('[')[1].split(']')[0] + 'xW' + df_key.split('[')[2].split(']')[0] 
        for df_key in df_keys
    }