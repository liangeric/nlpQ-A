'''
Class that will return a set of answers based on given question text
Command: ./answer article.txt questions.txt
Python command: python answer.py article.txt questions.txt
'''

from parse import Parse
import sys
import spacy
import numpy as np


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
            accum = np.zeros((300,))
            for word in sentence:

                # if given excludeTokens, skip word if it's in excludeTokens
                if excludeTokens is not None and word in excludeTokens:
                    continue

                if not word.is_stop:
                    accum += word.vector

            avg.append(accum / len(sentence))

        return avg

    def similarity(self):

        excludeTokens = set(["WHO", "WHAT", "WHEN", "WHERE", "HOW", "?"])

        qs = self.getAverageVector(self.spacyQuestions, excludeTokens)
        cs = self.getAverageVector(self.spacyCorpus)

        for i in range(len(qs)):
            cos = np.apply_along_axis(cosine, 1, cs, qs[i])

            print("Question:", list(self.spacyQuestions.sents)[i])
            print("Answer:", list(self.spacyCorpus.sents)[np.argmax(cos)])

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
