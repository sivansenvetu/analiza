import numpy as np
import matrix

# פונקציה ליצירת מטריצת אלמנטרית של הוספת שורות
def row_addition_elementary_matrix(n, target_row, source_row, scalar=1.0):
    # בדיקה אם אינדקסי השורות תקינים
    if target_row < 0 or source_row < 0 or target_row >= n or source_row >= n:
        raise ValueError("Invalid row indices.")

    # בדיקה אם השורות הן אותו הדבר
    if target_row == source_row:
        raise ValueError("Source and target rows cannot be the same.")

    # יצירת מטריצה אלמנטרית של הוספת שורות
    elementary_matrix = np.identity(n)
    elementary_matrix[target_row, source_row] = scalar

    return np.array(elementary_matrix)

# פונקציה ליצירת מטריצת אלמנטרית של החלפת שורות
def swap_rows_elementary_matrix(n, row1, row2):
    elementary_matrix = np.identity(n)
    # החלפת השורות row1 ו-row2
    elementary_matrix[[row1, row2]] = elementary_matrix[[row2, row1]]

    return np.array(elementary_matrix)

# פונקציה לחישוב פירוק LU של מטריצה
def lu(A):
    N = len(A)
    L = np.eye(N)  # יצירת מטריצה זהותית של גודל N x N

    for i in range(N):
        # חיפוש שורת הפיבוט עם הערך המוחלט הגדול ביותר בעמודה הנוכחית
        pivot_row = i
        v_max = A[pivot_row][i]
        for j in range(i + 1, N):
            if abs(A[j][i]) > v_max:
                v_max = A[j][i]
                pivot_row = j

        # בדיקה אם האלמנט האבחנתי הוא אפס, מה שיגרום לשגיאה בחלוקה מאוחרת
        if A[i][pivot_row] == 0:
            raise ValueError("can't perform LU Decomposition")

        # החלפת השורה הנוכחית עם שורת הפיבוט
        if pivot_row != i:
            e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
            print(f"elementary matrix for swap between row {i} to row {pivot_row} :\n {e_matrix} \n")
            A = np.dot(e_matrix, A)
            print(f"The matrix after elementary operation :\n {A}")

        # חישוב המונה להעלמת האלמנטים מתחת לפיבוט בעמודה הנוכחית
        for j in range(i + 1, N):
            m = -A[j][i] / A[i][i]
            e_matrix = row_addition_elementary_matrix(N, j, i, m)
            e_inverse = np.linalg.inv(e_matrix)
            L = np.dot(L, e_inverse)
            A = np.dot(e_matrix, A)
            print(f"elementary matrix to zero the element in row {j} below the pivot in column {i} :\n {e_matrix} \n")
            print(f"The matrix after elementary operation :\n {A}")

    U = A
    return L, U

# פונקציה לחישוב פתרונות מערכת משוואות באמצעות חישוב אחורה
def backward_substitution(U, b):
    N = len(U)
    x = np.zeros(N)  # יצירת מערך לאחסון הפתרונות

    # שילוב U ו-b למטריצה מורחבת
    augmented_matrix = np.hstack([U, b.reshape(-1, 1)])

    # חישוב אחורה מהמשוואה האחרונה ועד הראשונה
    for i in range(N - 1, -1, -1):
        x[i] = augmented_matrix[i, N]  # הערך של b[i]

        # חישוב הערך של x[i]
        for j in range(i + 1, N):
            x[i] -= U[i][j] * x[j]

        x[i] /= U[i][i]

    return x

# פונקציה לפתרון מערכת משוואות בעזרת פירוק LU
def lu_solve(A, b):
    L, U = lu(A)
    print("Lower triangular matrix L:\n", L)
    print("Upper triangular matrix U:\n", U)
    result = backward_substitution(U, b)
    print("\nVector X:")
    for x in result:
        print("{:.6f}".format(x))

# בדיקה עם מטריצה ווקטור לדוגמה
if __name__ == '__main__':
    A = np.array([[3, 2, 1],
                  [1, 4, -3],
                  [-2, 1, 5]])

    b = np.array([[1],
                  [1],
                  [1]])
    A_inverse = matrix.inverse(A)
    print("\nסעיף א:")
    print("Inverse of matrix A: \n", A_inverse)

    print("\nסעיף ב:")
    lu_solve(A, b)
