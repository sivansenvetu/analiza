import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import misc


def richardson(f, x, n, h):
    """Calculate Richardson's Extrapolation to approximate f'(x)."""
    d = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        d[i, 0] = (f(x + h) - f(x - h)) / (2 * h)
        for j in range(1, i + 1):
            d[i, j] = d[i, j-1] + (d[i, j-1] - d[i-1, j-1]) / (4**j - 1)
        h /= 2

    return d


def plot_convergence(results, true_derivative):
    """Plot the convergence graph of Richardson's Extrapolation."""
    plt.figure(figsize=(12, 6))
    for j in range(results.shape[1]):
        plt.plot(range(results.shape[0]), results[:, j], marker='o', label=f'Level {j}')

    plt.xlabel('Step')
    plt.ylabel('Estimated Derivative')
    plt.title("Convergence of Richardson's Extrapolation (Zoomed)")
    plt.legend()
    plt.grid(True)

    y_min = min(np.min(results), true_derivative) * 0.9999
    y_max = max(np.max(results), true_derivative) * 1.0001
    plt.ylim(y_min, y_max)

    plt.axhline(y=true_derivative, color='r', linestyle='--', label='True Derivative')
    plt.legend()

    plt.show()


def print_results_table(results):
    """Print the results table of Richardson's Extrapolation."""
    headers = [f"Level {i}" for i in range(results.shape[1])]
    table = tabulate(results, headers=headers, tablefmt="grid", floatfmt=".8f")
    print("Richardson Extrapolation Results:")
    print(table)


def analyze_results(results, true_derivative, x, n):
    """Analyze and print the results of Richardson's Extrapolation."""
    print("\nAnalysis:")
    print(f"1. The true value of the derivative at x={x} is {true_derivative:.8f}.")
    print(f"2. The initial estimate (Level 0) starts at {results[0, 0]:.8f} and approaches {true_derivative:.8f} as the step size decreases.")
    print("3. Each additional level of extrapolation provides a more accurate estimate.")
    print(f"4. At the highest level (Level {n}), we reach an accuracy of {results[n, n]:.8f}.")
    print("5. The graph shows how each level of extrapolation converges faster to the true value.")


def analyze_richardson(f, x, n, h):
    """Perform and analyze Richardson's Extrapolation."""
    results = richardson(f, x, n, h)
    true_derivative = misc.derivative(f, x, dx=1e-6)
    #true_derivative = 50
    print_results_table(results)
    plot_convergence(results, true_derivative)
    analyze_results(results, true_derivative, x, n)


def f(x):
    """Example function: f(x) = x^8."""
    return x**8


if __name__ == "__main__":
    """f : The function for which the derivative is being approximated.
    x : The point at which to compute the derivative.
    n : The number of extrapolation levels (the number of iterations to improve the accuracy).
    h : The initial step size used for the finite difference approximation."""
    x, n, h = 2, 5, 0.25
    analyze_richardson(f, x, n, h)