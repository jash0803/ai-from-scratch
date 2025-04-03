import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.variance = None

    def fit(self, X):
        # Step 1: Standardize the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Step 2: Compute covariance matrix
        covariance_matrix = np.cov(X.T)

        # Step 3: Calculate the Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvectors = eigenvectors.T

        # Step 4: Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1] # Sort in descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[sorted_indices]

        # Step 5: Select the top k eigenvectors
        self.components = eigenvectors[:, self.n_components]
        self.variance = eigenvalues[:self.n_components]
        return self
    
    def transform(self, X):
        # Step 6: Project the data onto the new feature space
        X = X - self.mean
        return np.dot(X, self.components.T)