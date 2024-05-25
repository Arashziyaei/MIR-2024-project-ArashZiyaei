import json

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ..word_embedding.fasttext_model import preprocess_text
from ..word_embedding.fasttext_model import FastText


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = FastText()
        self.review_tokens = []
        self.sentiments = []
        self.reviews = []
        self.embeddings = []

    def load_data(self, model_kind=''):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        df = pd.read_csv(self.file_path)
        df['review'] = df['review'].apply(lambda x: preprocess_text(text=x, minimum_length=1, stopword_removal=True,
                                                                    lower_case=True, punctuation_removal=True))
        self.reviews = df['review'].tolist()
        self.sentiments = [1 if x == 'positive' else 0 for x in df['sentiment'].tolist()]

        self.fasttext_model.prepare(mode='load', dataset=None, save=False, path=f'./FastText_{model_kind}.bin')
        self.fasttext_model.train(self.reviews)

        with open('./sentiment_labels', 'w') as f:
            json.dump(list(self.sentiments), f)
        #
        # for review in self.reviews:
        #     tokens = review.split()
        #     for token in tokens:
        #         if token not in self.review_tokens:
        #             self.review_tokens.append(token)
        with open('./normalized_tokens', 'r') as f:
            self.review_tokens = json.load(f)

        with open('./normalized_tokens', 'w') as f:
            json.dump(list(self.review_tokens), f)

        self.fasttext_model.prepare(mode='save', dataset=None, save=True, path=f'./FastText_{model_kind}.bin')

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        for review in self.reviews:
            self.embeddings.append(self.fasttext_model.get_query_embedding(review))

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.sentiments, test_size=test_data_ratio,
                                                            random_state=42)
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)