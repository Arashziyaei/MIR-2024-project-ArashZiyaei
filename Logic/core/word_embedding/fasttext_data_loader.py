import json

import numpy as np

from Logic.core.utility.preprocess import Preprocessor

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopsis, summaries, reviews, titles, genres).
        """
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        for i in range(len(data)):
            synopsis = ''
            if data[i]['synopsis'] is not None:
                synopsis = ' '.join(data[i]['synopsis'])
            data[i]['synopsis'] = synopsis

            summaries = ''
            if data[i]['summaries'] is not None:
                summaries = ' '.join(data[i]['summaries'])
            data[i]['summaries'] = summaries

            reviews = ''
            if data[i]['reviews'] is not None:
                for review, score in data[i]['reviews']:
                    reviews = reviews + review + ' '
            data[i]['reviews'] = reviews

        columns = ['synopsis', 'summaries', 'reviews', 'title', 'genres']
        df = pd.DataFrame(data, columns=columns)
        return df

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        unique_genres = set(genre for sublist in df['genres'] for genre in sublist)
        for genre in unique_genres:
            df[genre] = df['genres'].apply(lambda x: genre in x).astype(int)
        df = df.dropna(subset=df.columns[df.isnull().any()])
        columns = ['synopsis', 'summaries', 'reviews', 'title']
        x = []
        for row in df[columns]:
            st = ' '.join(row)
            x.append(st)
        y = df['genres'].values
        return np.array(x), y

    def create_train_data_for_cluster(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        unique_genres = set(genre for sublist in df['genres'] for genre in sublist)
        for genre in unique_genres:
            df[genre] = df['genres'].apply(lambda x: genre in x).astype(int)
        df = df.dropna(subset=df.columns[df.isnull().any()])
        x = df['summaries'].values
        y = df['genres'].values
        return x, y
