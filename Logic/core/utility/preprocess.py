import copy
import json

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation

class Preprocessor:

    def __init__(self, documents:list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(punctuation)
        self.documents = documents
        self.stopwords = []
        self.path_to_stop_words = '/Users/arashziyaei/sharif university/UNI/term6/MIR/project/MIR-2024-project-ArashZiyaei/Logic/core/utility/stopwords.txt'
        with open(self.path_to_stop_words, 'r') as file:
            self.stopwords = [word.strip() for word in file.readlines()]

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        preprocessed_docs = []
        for document in self.documents:
            if isinstance(document, str):
                document = self.preprocess_string(document)
            else:
                if document['first_page_summary'] is not None:
                    first_page_summary = document['first_page_summary']
                    document['first_page_summary'] = self.preprocess_string(first_page_summary)

                if document['genres'] is not None:
                    genres = document['genres']
                    document['genres'] = [self.preprocess_string(genre) for genre in genres]

                if document['stars'] is not None:
                    stars = document['stars']
                    document['stars'] = [self.preprocess_string(star) for star in stars]

                if document['summaries'] is not None:
                    summaries = document['summaries']
                    pr_summaries = []
                    for summary in summaries:
                        text = re.sub(r'<.*?>', '', summary)
                        pr_summary = self.preprocess_string(text)
                        pr_summaries.append(pr_summary)
                    document['summaries'] = pr_summaries

                if document['synopsis'] is not None:
                    synopsis = document['synopsis']
                    document['synopsis'] = [self.preprocess_string(s) for s in synopsis]

                if document['reviews'] is not None:
                    pr_reviews = []
                    doc_reviews = document['reviews']
                    for reviews in doc_reviews:
                        if reviews is not None:
                            preprocessed = self.preprocess_string(reviews[0])
                            pr_reviews.append([preprocessed, reviews[1]])
                    document['reviews'] = pr_reviews
            preprocessed_docs.append(document)
        return preprocessed_docs

    def preprocess_string(self, text: str):
        normalized_text = self.normalize(text)
        removed_links = self.remove_links(normalized_text)
        removed_punctuations = self.remove_punctuations(removed_links)
        preprocessed_string = ' '.join(self.remove_stopwords(removed_punctuations))
        return preprocessed_string

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        text = text.lower()
        tokens = self.tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        tokens = [self.stemmer.stem(token) for token in tokens]
        normalized_text = ' '.join(tokens)
        return normalized_text

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return ''.join([char for char in text if char not in self.punctuation])

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        tokens = self.tokenize(text)
        tokens = [token for token in tokens if
                  token.lower() not in self.stopwords and token not in self.punctuation]
        return tokens

if __name__ == '__main__':
    docs = None
    with open('../IMDB_crawled.json', 'r') as f:
        docs = json.load(f)
    print(len(docs))
    # TRUEDOCS = []
    # for doc in docs:
    #     d = {
    #         'id': doc['id'],  # str
    #         'title': doc['title'],  # str
    #         'first_page_summary': doc['first_page_summary'],  # str
    #         'release_year': doc['release_year'],  # str
    #         'mpaa': doc['mpaa'],  # str
    #         'budget': doc['budget'],  # str
    #         'gross_worldwide': doc['gross_worldwide'],  # str
    #         'rating': doc['rating'],  # str
    #         'directors': doc['directors'],  # List[str]
    #         'writers': doc['writers'],  # List[str]
    #         'stars': doc['stars'],  # List[str]
    #         'related_links': doc['related_links'],  # List[str]
    #         'genres': doc['genres'],  # List[str]
    #         'languages': doc['languages'],  # List[str]
    #         'countries_of_origin': doc['countries_of_origin'],  # List[str]
    #         'summaries': doc['summaries'],  # List[str]
    #         'synopsis': doc['synposis'],  # List[str]
    #         'reviews': doc['reviews'],  # List[List[str]]
    #     }
    #     print(d)
    #     TRUEDOCS.append(d)
    #
    # with open('../IMDB_crawled.json', 'w') as f:
    #     json.dump(TRUEDOCS, f)
    # print(len(TRUEDOCS))

    pre = Preprocessor(docs).preprocess()
    print(len(pre))
    with open('../preprocessed_documents.json', 'w') as f:
        json.dump(list(pre), f)