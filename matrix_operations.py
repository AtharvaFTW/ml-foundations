def multiply_matrix(A,B):
    """
    Multiplies matrix A (m x n) by matrix (n x p).
    Returns matrix C (m x p)
    """
    
    m = len(A)
    n = len(A[0])
    test = len(B)
    p = len(B[0])

    if not test == n:
        raise ValueError("The inner dimentions of matrix A and B must match.")

    C = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C


A = [[2, 1, 1],
     [3, 4, 2]]

B = [[1,  4],
     [2,  5],
     [3, 6]]
result = multiply_matrix(A, B)
print(result)
