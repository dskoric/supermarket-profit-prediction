import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../src'))
from linear_regression import compute_cost, compute_gradient

def test_compute_cost():
    x_test = np.array([1, 2, 3])
    y_test = np.array([1, 2, 3])
    w, b = 1, 0
    cost = compute_cost(x_test, y_test, w, b)
    assert cost == 0, "Cost should be zero for perfect fit"

def test_compute_gradient():
    x_test = np.array([1, 2, 3])
    y_test = np.array([1, 2, 3])
    w, b = 1, 0
    dj_dw, dj_db = compute_gradient(x_test, y_test, w, b)
    assert dj_dw == 0, "Gradient should be zero for perfect fit"
    assert dj_db == 0, "Gradient should be zero for perfect fit"