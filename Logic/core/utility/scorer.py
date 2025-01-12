import numpy as np

class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents
        self.raw_collection_size = 0
        for t, items in self.index.items():
            self.raw_collection_size += sum(items.values())

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            # TODO
            dft = len(self.index.get(term, {}).keys())
            idf = np.log10(self.N / (dft+1))
            self.idf[term] = idf
        return idf
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        
        #TODO
        query_tfs = {}
        for term in query:
            query_tfs[term] = query.count(term)
        return query_tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        scores = {}
        methods = method.split(".")
        list_of_docs = self.get_list_of_documents(query)
        for document_id in list_of_docs:
            query_tfs = self.get_query_tfs(query)
            score = self.get_vector_space_model_score(query, query_tfs, document_id, methods[0], methods[1])
            scores[document_id] = score
        return scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        #TODO
        query_vector = []
        document_vector = []

        for term in query:
            query_tf, query_idf = query_tfs[term], 1
            if query_method[0] == 'l':
                query_tf = np.log(query_tf+1)
            if query_method[1] == 't':
                query_idf = self.get_idf(term)
            query_tf_idf = query_tf * query_idf
            query_vector.append(query_tf_idf)

            document_idf = 1
            # if term in self.index and document_id in self.index[term]:
            document_tf = self.index.get(term, {}).get(document_id, 0)
            if document_method[0] == 'l':
                document_tf = np.log(document_tf+1)
            if document_method[1] == 't':
                document_idf = self.get_idf(term)
            document_tf_idf = document_tf * document_idf
            document_vector.append(document_tf_idf)

        query_norm = np.linalg.norm(np.array(query_vector))
        document_norm = np.linalg.norm(np.array(document_vector))

        if query_method[2] == 'c':
            query_vector /= query_norm

        if document_method[2] == 'c':
            document_vector /= document_norm

        vector_space_model_score = np.dot(query_vector, document_vector)
        return vector_space_model_score

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        scores = {}
        for document_id in self.get_list_of_documents(query):
            score = self.get_okapi_bm25_score(query, document_id, average_document_field_length, document_lengths)
            scores[document_id] = score
        return scores

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        # TODO
        b = 0.75
        k = 1.2
        score = 0
        for term in query:
            tf_i = self.index.get(term, {}).get(document_id, 0)
            B_component = (1 - b) + (b * document_lengths.get(document_id,0)) / average_document_field_length
            score += (self.get_idf(term) * tf_i * (k + 1)) / ((k * B_component) + tf_i)
        return score

    def compute_scores_with_unigram_model(
            self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """
        # TODO
        unigram_scores = {}
        for document_id in self.get_list_of_documents(query):
            score = self.compute_score_with_unigram_model(query, document_id, smoothing_method,
                                                          document_lengths, alpha, lamda)
            unigram_scores[document_id] = score
        return unigram_scores

    def compute_score_with_unigram_model(
            self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """
        # TODO
        corpus_tfs = {}
        for term in query:
            # corpus_tfs[term] = 0
            # if term in self.index:
            corpus_tfs[term] = sum(self.index.get(term, {}).values())
        score = 0
        document_length = document_lengths.get(document_id, 1e10)
        for term in query:
            # document_tf = 1
            # if term in self.index and document_id in self.index[term]:
            document_tf = self.index.get(term, {}).get(document_id, 0)
            if smoothing_method == 'naive':
                score += np.log((document_tf / document_length)+1)
            elif smoothing_method == 'bayes':
                corpus_tf = corpus_tfs[term]
                score += np.log(1+(document_tf + alpha * (corpus_tf / self.raw_collection_size)) / (document_length + alpha))
            else:
                corpus_tf = corpus_tfs[term]
                score += np.log(lamda * (document_tf / document_length) +
                                  (1 - lamda) * (corpus_tf / self.raw_collection_size))
        return score