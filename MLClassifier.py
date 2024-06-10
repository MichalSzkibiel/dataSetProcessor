import numpy as np

# This code comes from here: https://www.kaggle.com/code/dorianlazar/maximum-likelihood-classification-heart-disease.
class MLClassifier:
    def __init__(self):
        self.d = None
        self.nclasses = None
        self.mu_list = None
        self.sigma_inv_list = None
        self.scalars = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n)
        """
        # no. of variables / dimension
        self.d = x.shape[1]

        # no. of classes; assumes labels to be integers from 0 to nclasses-1
        self.nclasses = len(set(y))

        # list of means; mu_list[i] is mean vector for label i
        self.mu_list = []

        # list of inverse covariance matrices;
        # sigma_list[i] is inverse covariance matrix for label i
        # for efficiency reasons we store only the inverses
        self.sigma_inv_list = []

        # list of scalars in front of e^...
        self.scalars = []

        n = x.shape[0]
        for i in range(self.nclasses):

            # subset of obesrvations for label i
            cls_x = np.array([x[j] for j in range(n) if y[j] == i])

            mu = np.mean(cls_x, axis=0)

            # rowvar = False, this is to use columns as variables instead of rows
            sigma = np.cov(cls_x, rowvar=False)
            if np.sum(np.linalg.eigvals(sigma) <= 0) != 0:
                # if at least one eigenvalue is <= 0 show warning
                print(f'Warning! Covariance matrix for label {cls_x} is not positive definite!\n')

            sigma_inv = np.linalg.inv(sigma)

            scalar = 1 / np.sqrt(((2 * np.pi) ** self.d) * np.linalg.det(sigma))

            self.mu_list.append(mu)
            self.sigma_inv_list.append(sigma_inv)
            self.scalars.append(scalar)

    def _class_likelihood(self, x: np.ndarray, cls: int) -> float:
        """
        x - numpy array of shape (d),
        cls - class label

        Returns: likelihood of x under the assumption that class label is cls
        """
        mu = self.mu_list[cls]
        sigma_inv = self.sigma_inv_list[cls]
        scalar = self.scalars[cls]

        exp = (-1 / 2) * np.dot(np.matmul(x - mu, sigma_inv), (x - mu).T)

        return scalar * (np.e ** exp)

    def predict(self, x: np.ndarray) -> int:
        """
        x - numpy array of shape (d),
        Returns: predicted label
        """
        likelihoods = np.array([[self._class_likelihood(row, i) for i in range(self.nclasses)] for row in x])
        print(likelihoods.shape)
        return np.argmax(likelihoods, 1)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n),
        Returns: accuracy of predictions
        """
        n = x.shape[0]
        predicted_y = np.array([self.predict(x[i]) for i in range(n)])
        n_correct = np.sum(predicted_y == y)
        return n_correct / n
