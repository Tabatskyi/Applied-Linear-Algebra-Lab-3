import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt


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
# U, sigma, Vt = what_a_sigma(random_matrix)
# print(random_matrix)
# print(U, sigma, Vt)

file_path = 'ratings-small.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
print(ratings_matrix)

ratings_matrix = ratings_matrix.dropna(thresh=10, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=20, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)
U = np.array(U)
sigma = np.diag(sigma)
Vt = np.array(Vt)
V = Vt.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U[:20, 0], U[:20, 1], U[:20, 2])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V[:20, 0], V[:20, 1], V[:20, 2])
plt.show()

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
print(preds_df)
