import numpy as np

class NaiveBayes:

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Step 1: Calculate the mean and variance for each feature in each class
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.variance[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
        return self
    
    def predict(self, X):
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_samples):
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                class_conditional = np.sum(np.log(self._pdf(idx, X[i])))
                posterior = prior + class_conditional
                posteriors.append(posterior)
            y_pred[i] = self.classes[np.argmax(posteriors)]
        return y_pred
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        variance = self.variance[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator