import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


def what_a_sigma(matrix):
    AtA = np.dot(matrix.T, matrix)
    eigvals_AtA, eigvecs_AtA = np.linalg.eig(AtA)
    singular_values = np.sqrt(np.abs(eigvals_AtA))

    idx = np.argsort(singular_values)[::-1]
    singular_values = singular_values[idx]
    eigvecs_AtA = eigvecs_AtA[:, idx]

    sigma = np.diag(singular_values)
    Vt = eigvecs_AtA.T

    U = np.dot(matrix, Vt.T) / singular_values
    U = np.array([u / np.linalg.norm(u) if np.linalg.norm(u) > 0 else u for u in U.T]).T

    # if U.shape[1] > sigma.shape[0]:
    #     U = U[:, :sigma.shape[0]]

    if not np.allclose(matrix, U.dot(sigma).dot(Vt)):
        raise ValueError("SVD decomposition failed")

    return U, sigma, Vt


random_matrix = np.array(np.random.rand(10, 5))
U, sigma, Vt = what_a_sigma(random_matrix)
print(random_matrix)
print(U, sigma, Vt)
