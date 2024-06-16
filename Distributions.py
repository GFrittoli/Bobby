'''Utilities to handle distributions like Flagships: cutting, enhancing or adjusting y values, and finding indices around peaks.

Functions:
- adjust_y_values(y_array): Adjusts y values in the array.
- smooth_transition(x, x1, x2, a): Calculates the smooth transition for given x values.
- stepper(x, y, s, x_left, x_right): Applies a smooth transition to y based on specified x positions.
- cutter(x, y, thres): Cuts the array based on the gradient threshold.
- find_indices_around_peak(dndz_bins_column, initial_threshold=1e-4, max_threshold=1e-1): Finds indices around the peak based on the threshold.




'''

import numpy as np

def adjust_y_values(y_array):
    """
    Adjusts y values in the array: if the absolute value of y is 1e-5 or less,
    it multiplies that value by 100.

    Parameters:
    - y_array: NumPy array of y values.

    Returns:
    - adjusted_y: NumPy array with adjusted y values.
    """
    # Make a copy to avoid modifying the original array
    adjusted_y = np.copy(y_array)

    # Find indices where the absolute value of y is 1e-5 or less
    small_values_idx = np.abs(adjusted_y) <= 1e-5

    # Multiply those values by 100
    adjusted_y[small_values_idx] *= 1e5

    return adjusted_y



def smooth_transition(x, x1, x2, a):
    """
    Calculates the smooth transition for given x values between x1 and x2 with steepness a.

    Parameters:
    - x: array-like, the x values.
    - x1: scalar, start of the transition.
    - x2: scalar, end of the transition.
    - a: scalar, controls the steepness of the transition.

    Returns:
    - Array of same shape as x, representing the smooth transition.
    """
    return 0.5 * (1 + np.tanh(a * (x - x1))) * 0.5 * (1 - np.tanh(a * (x - x2)))

def stepper(x, y, s, x_left, x_right):
    """
    Applies a smooth transition to y based on specified x positions for the transition start and end.

    Parameters:
    - x: array-like, the x values.
    - y: array-like, the y values to be modified.
    - s: scalar, controls the steepness of the transition.
    - x_left: scalar, the x position to start the transition.
    - x_right: scalar, the x position to end the transition.

    Returns:
    - Array of same shape as y, representing the modified y values after applying the transition.
    """
    y2 = smooth_transition(x, x_left, x_right, s) * y

    return np.abs(y2)


def cutter(x,y,thres):
    idx = np.where(np.abs(np.gradient(y))>thres)[0]
    
    y2 = y[idx]
    x2 = x[idx]

    return x2, np.abs(y2)




def find_indices_around_peak(dndz_bins_column, initial_threshold=1e-4, max_threshold=1e-1):
    """
    Finds the indices of the first value before and after the peak that are of the same order of magnitude as the threshold.
    Automatically increases the threshold by an order of magnitude if one of the indices is None, up to a maximum threshold.

    Parameters:
    - dndz_bins_column: array-like, a specific column from a dataset representing distribution values.
    - initial_threshold: float, the starting order of magnitude to compare against (default is 1e-4).
    - max_threshold: float, the maximum threshold to escalate to (default is 1e-1).

    Returns:
    - A tuple containing the indices of the first value before and after the peak that match the order of magnitude of the threshold.
      If no valid indices are found within the max threshold, returns (None, None).
    """
    threshold = initial_threshold

    while threshold <= max_threshold:
        # Find the index of the maximum value in the column
        peak_index = np.argmax(dndz_bins_column)

        # Find the order of magnitude of the threshold
        order_of_magnitude = np.floor(np.log10(threshold))

        # Initialize indices
        before_index = None
        after_index = None

        # Search before the peak
        for i in range(peak_index, -1, -1):  # Iterate backwards from the peak to the start of the array
            if dndz_bins_column[i] <= 0:
                print(f"Value at index {i} is non-positive and ignored: {dndz_bins_column[i]}")
            elif np.floor(np.log10(dndz_bins_column[i])) == order_of_magnitude:
                before_index = i
                break

        # Search after the peak
        for i in range(peak_index, len(dndz_bins_column)):  # Iterate forwards from the peak to the end of the array
            if dndz_bins_column[i] <= 0:
                print(f"Value at index {i} is non-positive and ignored: {dndz_bins_column[i]}")
            elif np.floor(np.log10(dndz_bins_column[i])) == order_of_magnitude:
                after_index = i
                break

        # Check if both indices are found
        if before_index is not None and after_index is not None:
            return (before_index, after_index)

        # Escalate threshold by one order of magnitude
        print(f"Increasing threshold from {threshold} to {threshold*10} as indices not found.")
        threshold *= 10

    # Return None if no valid indices are found within the max threshold
    print("No valid indices found within the maximum threshold.")
    return (None, None)

# Example usage:
# Assuming 'dndz_example_column' is loaded as described in the commented section.
# indices = find_indices_around_peak(dndz_example_column)
# print("Indices around the peak:", indices)

