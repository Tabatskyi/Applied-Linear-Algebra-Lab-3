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

    if not np.allclose(matrix, U.dot(sigma).dot(Vt)):
        raise ValueError("SVD decomposition failed")

    return U, sigma, Vt


random_matrix = np.array(np.random.rand(10, 5))

ratings_file_path = 'ratings_small.csv'
movies_file_path = 'movies.csv'
ratings_df = pd.read_csv(ratings_file_path)
movies_df = pd.read_csv(movies_file_path)

ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=10, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=20, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(ratings_matrix.mean().mean())
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
ax.scatter(U[:, 0], U[:, 1], U[:, 2])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V[:, 0], V[:, 1], V[:, 2])
plt.show()

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)


def recommend_movies(preds_df, user_id, movies_df, original_ratings_matrix, num_recommendations=5):
    user_preds = preds_df.iloc[user_id]
    original_user_ratings = pd.DataFrame(original_ratings_matrix, columns=original_ratings_matrix.columns,
                                         index=original_ratings_matrix.index)
    non_rated_movies = original_user_ratings[original_user_ratings.isnull()].iloc[user_id]
    print(user_preds)
    print(non_rated_movies)

    recommendations = pd.concat([user_preds, non_rated_movies], axis=1, join='inner')
    recommendations = recommendations.iloc[:, 0]
    recommendations = recommendations.sort_values(ascending=False)
    recommendations = recommendations.head(num_recommendations)
    recommendations = recommendations.rename('predicted_rating')
    recommendations = recommendations.reset_index()
    recommendations = recommendations.merge(movies_df, on='movieId')
    return recommendations


user_id = int(input("Enter user ID for recommendations: "))
original_ratings_matrix = ratings_matrix
recommendations = recommend_movies(preds_df, user_id, movies_df, original_ratings_matrix, 5)

print("\nTop 5 movie recommendations for user:")
print(recommendations.iloc[:, 1:])
