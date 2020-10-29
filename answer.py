'''
Class that will return a set of answers based on given question text
Command: ./answer article.txt questions.txt
Python command: python answer.py article.txt questions.txt
'''

from parse import Parse
from sentence_transformers import SentenceTransformer # Pip installed
import sys
import spacy
import numpy as np
import re

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class Answer:
    def __init__(self, article, questions):
        self.article = article
        self.questions = questions

        self.nlp = spacy.load("en_core_web_md")  # spacy model

    def preprocess(self):
        """[preprocess the corpus and create spacy objects for corpus and questions]
        """

        p = Parse()

        self.corpus = p.parseCorpus(self.article)

        self.questions = re.sub(r"[\n]+", " ", self.questions)

        self.spacyCorpus = self.nlp(self.corpus)
        self.spacyQuestions = self.nlp(self.questions)

    def getAverageVector(self, doc, excludeTokens=None):
        """[get the average vectors for a given doc]

        Args:
            doc ([spacy]): [spacy model from text]
            excludeTokens ([set], optional): [tokens to exclude]. Defaults to None.

        Returns:
            [list]: [avg]
        """
        avg = []
        for sentence in doc.sents:
            accum = np.zeros((300,))  # This value is hardcoded from the Spacy Word2Vec model
            for word in sentence:

                # if given excludeTokens, skip word if it's in excludeTokens
                if excludeTokens is not None and word in excludeTokens:
                    continue

                if not word.is_stop:
                    accum += word.vector

            avg.append(accum / len(sentence))

        return avg

    def getSentenceVector(self, doc, excludeTokens=None):
        """[get the sentence vector for a given doc]

        Args:
            doc ([spacy]): [spacy model from text]
            excludeTokens ([set], optional): [tokens to exclude]. Defaults to None.

        Returns:
            [list]: list of Numpy Arrays of Sentence vector
        """
        # This is the only model I tried. First time running should cause a download but afterwards it doesnt download.
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')

        # Gonna build the question without question words and '?'
        sentences = []
        for sentence in doc.sents:
            parsedSentence = []
            for word in sentence:

                # if given excludeTokens, skip word if it's in excludeTokens
                if excludeTokens is not None and word in excludeTokens:
                    continue

                # I dont want to remove stopping since we are looking at the sentence level now
                parsedSentence.append(word.text)

            sentences.append(" ".join(parsedSentence)) # Now we join all of the word back together 

        # Takes in a list of strings, careful to feed in string and not spacy objects
        # Returns a list of numpy arrays
        sentence_embeddings = model.encode(sentences) 
        # assert(len(sentence_embeddings) == len(list(doc.sents)))
        return sentence_embeddings

    def similarity(self):

        excludeTokens = set(["WHO", "WHAT", "WHEN", "WHERE", "HOW", "?"])

        qs = self.getSentenceVector(self.spacyQuestions, excludeTokens)
        cs = self.getSentenceVector(self.spacyCorpus)

        for i in range(len(qs)):
            cos = np.apply_along_axis(cosine, 1, cs, qs[i])

            print("Question:", list(self.spacyQuestions.sents)[i])
            print("Answer:", list(self.spacyCorpus.sents)[np.argmax(cos)])
            print("")

    def categorize(self):
        pass

    def answerQuestion(self, question, sentence):
        pass


if __name__ == "__main__":
    article, questions = sys.argv[1], sys.argv[2]

    article = open(article, "r").read()
    questions = open(questions, "r").read()

    answer = Answer(article, questions)
    answer.preprocess()
    answer.similarity()
