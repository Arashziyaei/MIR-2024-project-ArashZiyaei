import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from Logic.core.word_embedding import preprocess_text


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    df = pd.read_csv('./IMDB Dataset.csv')
    labels = df['sentiment'].values
    df['review'] = df['review'].apply(lambda x: preprocess_text(text=x, minimum_length=1, stopword_removal=True,
                                                                lower_case=True, punctuation_removal=True))
    sentences = df['review'].values
    cv = CountVectorizer(max_features=350)
    X = cv.fit_transform(sentences).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    svm_classifier = SVMClassifier()
    svm_classifier.fit(X_train, y_train)

    report = svm_classifier.prediction_report(X_test, y_test)
    print(report)