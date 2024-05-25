import fasttext
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from string import punctuation

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()

    if punctuation_removal:
        text = ''.join([char for char in text if char not in punctuation])

    tokens = text.split()

    if stopword_removal:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [token for token in tokens if token not in stopwords_domain]

    tokens = [token for token in tokens if len(token) >= minimum_length]
    preprocessed_string = ' '.join(tokens)

    return preprocessed_string

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        preprocessed_texts = [preprocess_text(text) for text in tqdm(texts)]
        np.savetxt('./Train_data.txt', preprocessed_texts, fmt='%s')
        self.model = fasttext.train_unsupervised('./Train_data.txt', model=self.method)

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        preprocessed_query = preprocess_text(query)
        return self.model.get_sentence_vector(preprocessed_query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # Obtain word embeddings for the words in the analogy
        # TODO
        word1_vector = self.model[word1]
        word2_vector = self.model[word2]
        word3_vector = self.model[word3]

        # Perform vector arithmetic
        # TODO
        analogy_vector = word2_vector - word1_vector + word3_vector

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        # TODO
        # Exclude the input words from the possible results
        # TODO
        vectors = {word: self.model[word] for word in self.model.get_words() if word not in [word1, word2, word3]}

        # Find the word whose vector is closest to the result vector
        # TODO
        max_score = 0
        closest_word = ''
        for word, word_vector in vectors.items():
            score = np.dot(analogy_vector, word_vector) / np.linalg.norm(analogy_vector)
            if score > max_score:
                closest_word = word
        return closest_word

    def save_model(self, path='./FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='./FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)


if __name__ == "__main__":
    ft_model = FastText(method='skipgram')

    path = '../preprocessed_documents.json'
    ft_data_loader = FastTextDataLoader(path)

    X, y = ft_data_loader.create_train_data()

    ft_model.train(X)

    ft_model.prepare(dataset=None, save=True, mode="save")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "queen"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
