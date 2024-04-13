import json

import numpy as np
import itertools
import random
import copy

class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        with open('punctuations.txt', 'r') as file:
            punctuations = [word.strip() for word in file.readlines()]

        self.punctuations = punctuations
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        # TODO
        shingles = set()

        for p in self.punctuations:
            document = document.replace(p, '')

        document = document.lower()
        words = document.split(' ')
        for i in range(len(document) - k + 1):
            shingles.add(' '.join(words[i:i + k]))
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        # TODO
        all_shingles = []
        documents_shingles = []
        for document in self.documents:
            shingles = self.shingle_document(document)
            all_shingles.extend(shingles)
            documents_shingles.append(shingles)
        all_shingles = list(set(all_shingles))
        num_docs = len(self.documents)
        characteristic_matrix = np.zeros((len(all_shingles), num_docs))

        for i in range(len(all_shingles)):
            for j in range(num_docs):
                if all_shingles[i] in documents_shingles[j]:
                    characteristic_matrix[i][j] = 1

        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """

        # TODO
        signatures_matrix = []
        characteristic_matrix = self.build_characteristic_matrix()
        num_shingles, num_docs = characteristic_matrix.shape
        permutations = [np.random.permutation(num_shingles) for _ in range(self.num_hashes)]
        for i in range(self.num_hashes):
            permutation = list(permutations[i])
            check = [-1 for i in range(len(self.documents))]
            for j in range(num_shingles):
                index = permutation.index(j)
                if -1 in check:
                    for k in range(len(self.documents)):
                        if check[k] == -1 and characteristic_matrix[index][k] == 1:
                            check[k] = j
                else:
                    break
            signatures_matrix.append(check)
        return np.array(signatures_matrix)

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        lsh_buckets = {}
        num_documents = len(self.documents)
        for band_index in range(bands):
            hash_values = signature[band_index*rows_per_band:(band_index+1)*rows_per_band]
            band_bucket = [hash(tuple(hash_values[:, i])) for i in range(num_documents)]
            for doc_index in range(num_documents):
                bucket = band_bucket[doc_index]
                if bucket in lsh_buckets:
                    lsh_buckets[bucket].append(doc_index)
                else:
                    lsh_buckets[bucket] = [doc_index]
        return lsh_buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        signature = self.min_hash_signature()
        lsh_buckets = self.lsh_buckets(signature, bands=25, rows_per_band=4)
        return lsh_buckets

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        #TODO
        jaccard_score = 0
        if first_set and second_set:
            intersection = len(first_set.intersection(second_set))
            union = len(first_set.union(second_set))
            jaccard_score = intersection / union
        return jaccard_score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)
if __name__ == '__main__':
    crawled_files = []
    with open('crawled_files.json', 'r') as f:
        crawled_files = json.load(f)
    crawled_files = crawled_files[:200]

    data = None
    with open('LSHFakeData.json', 'r') as f:
        data = json.load(f)

    data.extend(crawled_files)
    all_documents = []
    for doc in data:
        if doc['summaries'] is not None:
             all_documents.append(' '.join(doc['summaries']))

    min_hash_lsh = MinHashLSH(all_documents, 100)
    min_hash_lsh.jaccard_similarity_test(min_hash_lsh.perform_lsh(), all_documents)