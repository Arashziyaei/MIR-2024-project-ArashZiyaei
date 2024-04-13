from index_reader import Index_reader
from indexes_enum import Indexes, Index_types
import json

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """

        #TODO
        self.path = path
        self.documents = {}
        self.read_documents()
        self.metadata_index = self.create_metadata_index()


    def read_documents(self):
        """
        Reads the documents.
        
        """

        #TODO
        index_reader = Index_reader('./index/', Indexes.DOCUMENTS)
        self.documents = index_reader.index


    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['average_document_length'] = {
            'stars': self.get_average_document_field_length('stars'),
            'genres': self.get_average_document_field_length('genres'),
            'summaries': self.get_average_document_field_length('summaries')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self, where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """

        #TODO
        count = 0
        for ID, doc in self.documents.items():
            if doc[where] is not None:
                for s in doc[where]:
                    count += len(s.split(" "))
        return count / len(self.documents)

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)
    
if __name__ == "__main__":
    meta_index = Metadata_index()
    meta_index.create_metadata_index()
    meta_index.store_metadata_index('./index/')