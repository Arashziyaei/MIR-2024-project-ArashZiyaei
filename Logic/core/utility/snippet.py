import copy
class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        stopwords = None
        with open('/Users/arashziyaei/sharif university/UNI/term6/MIR/project/MIR-2024-project-ArashZiyaei/Logic/core/utility/stopwords.txt', 'r') as file:
            stopwords = [word.strip() for word in file.readlines()]
        tokens = query.split(' ')
        tokens = [token for token in tokens if token.lower() not in stopwords]
        return ' '.join(tokens)

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.
        doc_to_lower = doc.lower()
        doc_words = doc_to_lower.split()

        text = self.remove_stop_words_from_query(query)
        text = text.lower()
        query_words = text.split()

        for w in query.split():
            if w.lower() not in doc_words:
                not_exist_words.append(w)

        query_set_words = set(query_words)
        current_block = []
        positions = []
        position = 0
        min_length = float('inf')
        best_block = []
        for word in doc_words:
            if word in query_words:
                current_block.append(word)
                positions.append(position)

                # if current block[0] is repeated, remove the first element
                while current_block.count(current_block[0]) > 1:
                    current_block.remove(current_block[0])
                    positions.remove(positions[0])

                has_all_words = True
                for w in query_set_words:
                    if w not in current_block:
                        has_all_words = False
                        break

                # if current block has all query words, keep the current block with min length
                if has_all_words:
                    current_block_length = positions[-1] - positions[0] + 1
                    if current_block_length < min_length:
                        min_length = current_block_length
                        best_block = positions.copy()

            position += 1
        doc_words = doc.split(" ")
        seen_words = set()

        for i in range(len(best_block)):
            if doc_words[best_block[i]] in seen_words:
                final_snippet = final_snippet + ' ' + doc_words[best_block[i]]
            else:
                seen_words.add(doc_words[best_block[i]])
                final_snippet = final_snippet + ' ' + "***" + doc_words[best_block[i]] + "***"

            if i == 0:
                before_words = doc_words[:best_block[i]]
                if 0 < len(before_words) <= 5:
                    final_snippet = ' '.join(before_words) + final_snippet
                elif len(before_words) > 5:
                    final_snippet = '...' + ' '.join(before_words[-5:]) + final_snippet

            if i == len(best_block) - 1:
                after_words = doc_words[best_block[i] + 1:]
                if 0 < len(after_words) <= 5:
                    final_snippet = final_snippet + ' ' + ' '.join(after_words)
                elif len(after_words) > 5:
                    final_snippet = final_snippet + ' ' + ' '.join(after_words[:5]) + '...'

            if i != len(best_block) - 1:
                between_words = doc_words[best_block[i] + 1:best_block[i + 1]]
                if len(between_words) > 10:
                    final_snippet = final_snippet + " " + ' '.join(between_words[:5]) + "..." + ' '.join(
                        between_words[-5:])
                else:
                    final_snippet = final_snippet + " " + ' '.join(between_words)

        return final_snippet, not_exist_words

if __name__ == '__main__':
    snippet = Snippet()
    query = "F B C"
    doc = "C x x x x x x x x x x x x x x x F x x x x x x x x x x " \
          "x x x x x x x x B x x x x x x x x x x x x x x x x x x x B x F x " \
          "x x x x x x x x x x x x x x x x x x B x C x x x x x x x x x x x x x x x x" \
          " x x x B x x x x x x x x x x x x x F x F x C"
    final_string, not_exist_words = snippet.find_snippet(doc, query)
    print(final_string)