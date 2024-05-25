import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.word_embedding import preprocess_text
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


# class NaiveBayes(BasicClassifier):
#     def __init__(self, count_vectorizer, alpha=1):
#         super().__init__()
#         self.cv = count_vectorizer
#         self.num_classes = None
#         self.classes = None
#         self.number_of_features = None
#         self.number_of_samples = None
#         self.prior = None
#         self.feature_probabilities = None
#         self.log_probs = None
#         self.alpha = alpha
#
#     def fit(self, x, y):
#         self.classes, class_counts = np.unique(y, return_counts=True)
#         self.num_classes = len(self.classes)
#         self.number_of_features = x.shape[1]
#         self.number_of_samples = x.shape[0]
#
#         self.prior = class_counts / self.number_of_samples
#         self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
#
#         for idx, c in enumerate(self.classes):
#             x_class = x[y == c]
#             class_feature_counts = np.sum(x_class, axis=0) + self.alpha
#             self.feature_probabilities[idx, :] = class_feature_counts / (
#                         class_feature_counts.sum() + self.alpha * self.number_of_features)
#
#         self.log_probs = np.log(self.feature_probabilities)
#         return self
#
#     def predict(self, x):
#         log_prior = np.log(self.prior)
#         log_likelihood = x @ self.log_probs.T
#         log_posterior = log_likelihood + log_prior
#         return self.classes[np.argmax(log_posterior, axis=1)]
#
#     def prediction_report(self, x, y):
#         y_pred = self.predict(x)
#         return classification_report(y, y_pred)
#
#     def get_percent_of_positive_reviews(self, sentences):
#         x = self.cv.transform(sentences)
#         predictions = self.predict(x)
#         positive_count = np.sum(predictions == 1)
#         return (positive_count / len(sentences)) * 100
#
#
# # F1 Accuracy : 85%
# if __name__ == '__main__':
#     # Example usage (assuming ReviewLoader and BasicClassifier are correctly implemented)
#     df = pd.read_csv('./IMDB Dataset.csv')
#     df['review'] = df['review'].apply(lambda x: preprocess_text(text=x, minimum_length=1, stopword_removal=True,
#                                                                 lower_case=True, punctuation_removal=True))
#     sentences = np.array(df['review'].values)
#     labels = np.array(df['sentiment'].values)
#
#     cv = CountVectorizer(max_features=300)
#     X = cv.fit_transform(sentences).toarray()
#     X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
#
#     nb = NaiveBayes(count_vectorizer=cv)
#     nb.fit(X_train, y_train)
#
#     report = nb.prediction_report(X_test, y_test)
#     print(report)

class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        '''
        Initialize these:

        Laplace smoothing parameter
        P(Y)
        P(X_i | Y)
        '''
        super().__init__()
        self.alpha = alpha
        self.cv = count_vectorizer
        self.class_probabilities = {}  # prob of classes
        self.word_probabilities = {}  # for each word  = [prob. positive, negative]
    # def __init__(self, count_vectorizer, alpha=1):
    #     super().__init__()
    #     self.cv = count_vectorizer
    #     self.num_classes = None
    #     self.classes = None
    #     self.number_of_features = None
    #     self.number_of_samples = None
    #     self.prior = None
    #     self.feature_probabilities = {}
    #     self.log_probs = None
    #     self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        # self.classes, class_counts = np.unique(y, return_counts=True)
        # self.num_classes = len(self.classes)
        # self.number_of_features = x.shape[1]
        # self.number_of_samples = x.shape[0]
        #
        # self.prior = class_counts / self.number_of_samples
        # self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        #
        # for idx, c in enumerate(self.classes):
        #     x_class = x[y == c]
        #     class_feature_counts = np.sum(x_class, axis=0) + self.alpha
        #     self.feature_probabilities[idx, :] = class_feature_counts / (
        #                 class_feature_counts.sum() + self.alpha * self.number_of_features)
        #
        # self.log_probs = np.log(self.feature_probabilities)
        # return self
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        for i, class_ in enumerate(classes):
            self.class_probabilities[class_] = counts[i] / total_samples
        num_rows = len(y)
        num_columns = len(X[0])
        for i in range(num_columns):
            self.word_probabilities[i] = []
            word_related = []
            for j in range(num_rows):
                word_related.append(X[j][i])
            related_index = [j for j in range(num_rows) if word_related[j] == 1]

            positive = 0
            negative = 0

            for j in related_index:
                if y[j] == 'positive':
                    positive += word_related[j]
                else:
                    negative += word_related[j]

            total = len(related_index)
            positive_prob = (positive + self.alpha) / (total + self.alpha * X.shape[1])
            negative_prob = (negative + self.alpha) / (total + self.alpha * X.shape[1])

            self.word_probabilities[i].append(positive_prob)
            self.word_probabilities[i].append(negative_prob)

        return self

    def predict(self, X):
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
        for test in X:
            likelihood = float("-inf")
            predicted_class = None
            for class_, class_prob in self.class_probabilities.items():
                index = 1
                if class_ == 'positive':
                    index = 0
                probs_in_words = []
                for i in self.word_probabilities.items():
                    probs_in_words.append(i[1][index])
                score = np.log(class_prob)
                for i in range(len(test)):
                    if test[i] == 1:
                        score += np.log(probs_in_words[i])
                if score > likelihood:
                    likelihood = score
                    predicted_class = class_
            predictions.append(predicted_class)
        return predictions

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
        prediction = self.predict(x)
        return classification_report(y, prediction)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences)
        predictions = self.predict(x)
        positive_count = np.sum(predictions == 1)
        return (positive_count / len(sentences)) * 100


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the reviews using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    df = pd.read_csv('./IMDB Dataset.csv')
    df['review'] = df['review'].apply(lambda x: preprocess_text(text=x, minimum_length=1, stopword_removal=True,
                                                                lower_case=True, punctuation_removal=True))
    sentences = np.array(df['review'].values)
    labels = np.array(df['sentiment'].values)
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
    cv = CountVectorizer(max_features=3500)
    embeddings_count_train = cv.fit_transform(X_train).toarray()
    embeddings_count_test = cv.transform(X_test).toarray()

    naive_bayes = NaiveBayes(count_vectorizer=cv)
    naive_bayes.fit(embeddings_count_train, y_train)
    report = naive_bayes.prediction_report(embeddings_count_test, y_test)
    print(report)

