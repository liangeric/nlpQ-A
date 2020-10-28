'''
Class that will return a set of questions based on given text
Command: ./ask article.txt nquestions
Python command: python answer.py article.txt 21
'''

import sys


class Ask:
    def __init__(self, article, nquestions):
        self.article = article
        self.nquestions = nquestions

    def preprocess(self):
        pass

    def generateQuestions(self):
        pass

    def chooseNQuestions(self):
        pass


if __name__ == "__main__":
    article, nquestions = sys.argv[1], sys.argv[2]

    article = open(article, "r").read()
    nquestions = int(nquestions)

    ask = Ask(article, nquestions)
    ask.preprocess()
    ask.generateQuestions()
    ask.chooseNQuestions()
