import os
import numpy as np

def load_data():
    """
    Load dataset from file
    Returns:
        x (ndarray): Input features
        y (ndarray): Target values
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root and then to the data directory
    data_path = os.path.join(current_dir, '..', 'data', 'ex1data1.txt')
    
    data = np.loadtxt(data_path, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y