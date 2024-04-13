import time
import os
import json
import copy
import nltk
from Logic.core.preprocess import Preprocessor
from indexes_enum import Indexes

class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """
        #         TODO
        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc['id']] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        for doc in self.preprocessed_documents:
            stars = doc['stars']
            if stars is not None:
                for star in stars:
                    terms = star.split(' ')
                    for term in terms:
                        if term not in current_index:
                            current_index[term] = {}
                        count = 0
                        for s in stars:
                            count += s.count(term)
                        current_index[term][doc['id']] = count
        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        for doc in self.preprocessed_documents:
            genres = doc['genres']
            if genres is not None:
                for genre in genres:
                    if genre not in current_index:
                        current_index[genre] = {}
                    current_index[genre][doc['id']] = 1
        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        #         TODO
        for doc in self.preprocessed_documents:
            summaries = doc['summaries']
            if summaries is not None:
                for summary in summaries:
                    terms = summary.split(' ')
                    for term in terms:
                        if term not in current_index:
                            current_index[term] = {}
                        count = 0
                        for s in summaries:
                            count += s.count(term)
                        current_index[term][doc['id']] = count
        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            index = self.index.get(index_type)
            return index.get(word, [])

        except Exception as e:
            print("Error: ", e)
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        #         TODO
        # update document index
        self.index[Indexes.DOCUMENTS.value][document['id']] = document

        # update stars index
        stars = document['stars']
        if stars is not None:
            for star in stars:
                terms = star.split(' ')
                for term in terms:
                    if term not in self.index[Indexes.STARS.value]:
                        self.index[Indexes.STARS.value][term] = {}
                    count = 0
                    for s in stars:
                        count += s.count(term)
                    self.index[Indexes.STARS.value][term][document['id']] = count

        # update genres index
        genres = document['genres']
        if genres is not None:
            for genre in genres:
                if genre not in self.index[Indexes.GENRES.value]:
                    self.index[Indexes.GENRES.value][genre] = {}
                self.index[Indexes.GENRES.value][genre][document['id']] = 1

        # update summaries index
        summaries = document['summaries']
        if summaries is not None:
            for summary in summaries:
                terms = summary.split(' ')
                for term in terms:
                    if term not in self.index[Indexes.SUMMARIES.value]:
                        self.index[Indexes.SUMMARIES.value][term] = {}
                    count = 0
                    for s in summaries:
                        count += s.count(term)
                    self.index[Indexes.SUMMARIES.value][term][document['id']] = count

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        #         TODO
        # remove from document index
        if document_id in self.index[Indexes.DOCUMENTS.value]:
            del self.index[Indexes.DOCUMENTS.value][document_id]

        # remove from stars index
        for term, documents in self.index[Indexes.STARS.value].items():
            if document_id in documents:
                del self.index[Indexes.STARS.value][term][document_id]

        # remove from genres index
        for term, documents in self.index[Indexes.GENRES.value].items():
            if document_id in documents:
                del self.index[Indexes.GENRES.value][term][document_id]

        # remove from summaries index
        for term, documents in self.index[Indexes.SUMMARIES.value].items():
            if document_id in documents:
                del self.index[Indexes.SUMMARIES.value][term][document_id]

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """
        # important: my crawled data does not have start with name henry, so I used another name ('franca')
        dummy_document = {
            'id': '100',
            'stars': ['tim', 'franca'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['franca']).difference(set(index_before_add[Indexes.STARS.value]['franca']))
                != {dummy_document['id']}):
            print('Add is incorrect, franca')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """
        if index_name not in self.index:
            raise ValueError('Invalid index name')

        with open(path, 'w') as f:
            json.dump(self.index[index_name], f, indent=4)

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        # Path form: ./index/index_name.json
        #         TODO
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found at {path}")

        loaded_data = {}
        index_name = path.split('/')[1].split('.')[0]
        with open(path, 'r') as f:
            loaded_data = json.load(f)

        self.index[index_name] = loaded_data

        return loaded_data

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods
if __name__ == '__main__':
    pre_docs = {}

    with open('../preprocessed_documents.json', 'r') as f:
        pre_docs = json.load(f)
    print("check index:")
    index = Index(pre_docs)
    print("-----------------------------------------------------------------")
    print("1. Check load and store")
    print("1.1 documents:")
    # store and load documents
    index.store_index('./index/documents_index.json', 'documents')
    loaded_data = index.load_index('./index/documents_index.json')
    print(index.check_if_index_loaded_correctly('documents', loaded_data))

    print("1.2 stars:")
    # store and load stars
    index.store_index('./index/stars_index.json', 'stars')
    loaded_data = index.load_index('./index/stars_index.json')
    print(index.check_if_index_loaded_correctly('stars', loaded_data))

    print("1.3 genres:")
    # store and load genres
    index.store_index('./index/genres_index.json', 'genres')
    loaded_data = index.load_index('./index/genres_index.json')
    print(index.check_if_index_loaded_correctly('genres', loaded_data))

    print("1.4 summaries:")
    # store and load summaries
    index.store_index('./index/summaries_index.json', 'summaries')
    loaded_data = index.load_index('./index/summaries_index.json')
    print(index.check_if_index_loaded_correctly('summaries', loaded_data))

    print("-----------------------------------------------------------------")
    print("2. Check add and remove:")
    index.check_add_remove_is_correct()

    print("-----------------------------------------------------------------")
    print("3. check indexing is good:")
    print("3.1 documents:")
    index.check_if_indexing_is_good('documents', 'tt0119558')

    print("3.2 stars:")
    index.check_if_indexing_is_good('stars', 'leonardo')

    print("3.3 genres:")
    index.check_if_indexing_is_good('genres', 'crime')

    print("3.4 summaries:")
    index.check_if_indexing_is_good('summaries', 'good')
    print("-----------------------------------------------------------------")