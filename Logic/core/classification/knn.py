import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.X_train = x
        self.y_train = y
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = []
        for d in tqdm(x):
            prediction = self._predict(d)
            predictions.append(prediction)

        return np.array(predictions)

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            distance = self.euclidean_distance(x, x_train)
            distances.append(distance)
        distances = np.array(distances)

        n_neighbors_idxs = np.argsort(distances)[: self.k]

        labels = self.y_train[n_neighbors_idxs]
        labels = list(labels)
        return labels[0]

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        predictions = self.predict(x)
        return classification_report(y, predictions)


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    review_loader = ReviewLoader(file_path='./IMDB Dataset.csv')
    review_loader.load_data(model_kind='knn')
    review_loader.get_embeddings()
    X_train, X_test, y_train, y_test = review_loader.split_data()

    knn = KnnClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    report = knn.prediction_report(X_test, y_test)
    print(report)
