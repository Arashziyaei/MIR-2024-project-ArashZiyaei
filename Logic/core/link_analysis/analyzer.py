import json
import random

from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = {}
        self.authorities = {}
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            #TODO
            self.graph.add_node(movie['id'])
            if movie['stars'] is not None:
                for star in movie['stars']:
                    self.graph.add_node(star)
                    self.graph.add_edge(star, movie['id'])


    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            #TODO
            stars = movie['stars']
            if stars is not None:
                for item in self.root_set:
                    if item['stars'] is not None:
                        for star in stars:
                            if star in item['stars']:
                                self.graph.add_node(star)
                                self.graph.add_edge(star, item['id'])
                                self.graph.add_edge(star, movie['id'])

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = []
        h_s = []
        # TODO

        for node in self.graph.graph.nodes():
            self.authorities[node] = 1
            self.hubs[node] = 1

        for _ in range(num_iteration):
            new_auth = {}
            new_hubs = {}

            for node in self.graph.graph.nodes():
                auth_score = 0
                for pred in self.graph.get_predecessors(node):
                    auth_score += self.hubs[pred]
                new_auth[node] = auth_score

            for node in self.graph.graph.nodes():
                hub_score = 0
                for suc in self.graph.get_successors(node):
                    hub_score += new_auth[suc]
                new_hubs[node] = hub_score

            norm = sum(new_auth.values()) ** 0.5
            for node in self.graph.graph.nodes():
                new_auth[node] /= norm

            norm = sum(new_hubs.values()) ** 0.5
            for node in self.graph.graph.nodes():
                new_hubs[node] /= norm

            self.authorities = new_auth
            self.hubs = new_hubs

        a_s = sorted(self.hubs.items(), key=lambda x: x[1], reverse=True)[:max_result]
        h_s = sorted(self.authorities.items(), key=lambda x: x[1], reverse=True)[:max_result]
        return a_s, h_s

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    corpus = []    # TODO: it shoud be your crawled data
    with open('../preprocessed_documents.json', 'r') as f:
        corpus = json.load(f)

    # root_set = random.sample(corpus, 100)   # TODO: it shoud be a subset of your corpus
    root_set = corpus[:100]
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
