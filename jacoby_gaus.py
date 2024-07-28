import numpy as np

n = 3


def diagonally_dominant(A):
    for i in range(n):
        diagonal_element = abs(A[i][i])
        row_sum = 0
        for j in range(n):
            if i != j:
                row_sum += abs(A[i][j])
        if diagonal_element <= row_sum:
            return False
    return True


def jacobi(A, b, tolerance=0.001):
    x = np.zeros_like(b, dtype=np.double)
    x_new = np.zeros_like(b, dtype=np.double)
    while True:
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.allclose(x, x_new, atol=tolerance, rtol=0.):
            return x_new
        x = np.copy(x_new)


def gauss_seidel(A, b, stop_cond=0.001):
    x = np.zeros_like(b, dtype=np.double)
    while True:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.allclose(x, x_new, atol=stop_cond, rtol=0.):
            return x_new
        x = np.copy(x_new)


def main():
    A = np.array([[4, 2, 0],
                  [2, 10, 4],
                  [0, 4, 5]], dtype=np.double)
    b = np.array([5, 5, 5], dtype=np.double)

    if diagonally_dominant(A):
        print("Jacobi solution:")
        print(jacobi(A, b))

        print("\nGauss-Zeidel solution:")
        print(gauss_seidel(A, b))
    else:
        print("The matrix is not diagonally dominant.")


if __name__ == "__main__":
    main()
