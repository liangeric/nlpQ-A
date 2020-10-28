'''
Class that takes care of parsing any necessary files
'''

import re


class Parse:

    def parseCorpus(self, corpus):
        """[parse the given corpus according to certain rules]

        Args:
            corpus ([str]): [given article corpus]

        Returns:
            [str]: [new cleaned corpus]
        """
        corpus = corpus.strip()

        # convert to lower case, might not want to do this for parsing later
        # corpus = corpus.lower()

        # convert multiple new lines into a single new line
        corpus = re.sub(r"[\n]+", "\n", corpus)

        # convert periods with a bunch of spaces/newlines/tabs after to just a period
        # however does not convert if the period and spaces/newlines/tabs is followed by the ) character
        # this helps not filter out cases where the sentence is like (c. 2686-2181 BC)
        #corpus = re.sub(r"\.([\s\n\t]+)(?![^\.\(]*\))",".",corpus)

        # remove headers (assuming header's don't have periods, question marks, or exclamation marks)
        corpus = corpus.split("\n")
        corpus = [x for x in corpus if "." in x or "?" in x or "!" in x]
        corpus = " ".join(corpus)

        # convert the new lines into space
        corpus = re.sub(r"[\n]+", " ", corpus)

        # convert the spaces and tabs into single space
        corpus = re.sub(r"[\s\t]+", " ", corpus)

        return corpus
