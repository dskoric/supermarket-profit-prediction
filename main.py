import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_data
from linear_regression import compute_cost, compute_gradient, gradient_descent

def run_analysis(population_to_predict=None, alpha=0.01, iterations=1500):
    """
    Run the complete linear regression analysis
    """
    print("Loading data...")
    x_train, y_train = load_data()
    
    print("Training model...")
    initial_w = 0.
    initial_b = 0.
    w, b, _, _ = gradient_descent(x_train, y_train, initial_w, initial_b, 
                                 compute_cost, compute_gradient, alpha, iterations)
    
    print(f"Model parameters: w = {w}, b = {b}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, marker='x', c='r', label='Training Data')
    
    # Plot linear fit
    x_line = np.linspace(min(x_train), max(x_train), 100)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, c='b', label='Linear Fit')
    
    plt.title("Profits vs. Population per city")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.legend()
    plt.savefig('results/linear_regression_fit.png')
    print("Plot saved to results/linear_regression_fit.png")
    
    # Make predictions if requested
    if population_to_predict:
        population = np.array(population_to_predict)
        predictions = w * population + b
        for pop, pred in zip(population, predictions):
            print(f"For population = {pop*10000}, we predict a profit of ${pred*10000:.2f}")
    
    return w, b

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Regression for Supermarket Profits')
    parser.add_argument('--predict', type=float, nargs='+', 
                        help='Population values to predict (in 10,000s)')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Learning rate for gradient descent')
    parser.add_argument('--iterations', type=int, default=1500,
                        help='Number of iterations for gradient descent')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run the analysis
    run_analysis(args.predict, args.alpha, args.iterations)