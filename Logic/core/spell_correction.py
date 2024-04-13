import json


class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        
        # TODO: Create shingle here
        for i in range(len(word) - k + 1):
            shingles.add(word[i:i + k])
        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        jaccard_score = 0
        if first_set and second_set:
            intersection = len(first_set.intersection(second_set))
            union = len(first_set.union(second_set))
            jaccard_score = intersection / union
        return jaccard_score

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        # TODO: Create shingled words dictionary and word counter dictionary here.
        for doc in all_documents:
            words = doc.split(" ")
            for word in words:
                if word not in all_shingled_words:
                    all_shingled_words[word] = self.shingle_word(word)
                if word not in word_counter:
                    word_counter[word] = 0
                word_counter[word] += 1
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        # TODO: Find 5 nearest candidates here.
        word_shingle_set = self.shingle_word(word)
        scores = {}
        for candidate, candidate_shingle_set in self.all_shingled_words.items():
            jaccard_score = self.jaccard_score(word_shingle_set, candidate_shingle_set)
            scores[candidate] = jaccard_score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top5_candidates = sorted_candidates[:5]
        return top5_candidates
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = []

        # TODO: Do spell correction here.
        query_words = query.split(' ')
        for word in query_words:
            if word in self.word_counter:
                final_result.append(word)
            else:
                top_five_candidates = self.find_nearest_words(word)
                final_scores = [(candidate, jaccard_score * self.word_counter[candidate]) for candidate, jaccard_score in top_five_candidates]
                final_word = max(final_scores, key=lambda x: x[1])[0]
                final_result.append(final_word)
        return ' '.join(final_result)

if __name__ == '__main__':
    pre_docs = {}
    with open('./preprocessed_documents.json', 'r') as f:
        pre_docs = json.load(f)
    docs = []
    for doc in pre_docs:
        if doc['summaries'] is not None:
            for s in doc['summaries']:
                docs.append(s)
    spell_correction = SpellCorrection(docs)
    print(spell_correction.spell_check('spidern'))