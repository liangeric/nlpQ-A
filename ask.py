'''
Class that will return a set of questions based on given text
Command: ./ask article.txt nquestions
Python command: python answer.py article.txt 21
'''

import sys

import numpy as np
import spacy

from parse import Parse


class Ask:
    def __init__(self, article, nquestions):
        self.article = article
        self.nquestions = nquestions
        self.nlp = spacy.load("en_core_web_md")  # spacy model
        self.preprocess()

    def preprocess(self):
        p = Parse()

        self.corpus = p.parseCorpus(self.article)
        self.spacyCorpus = self.nlp(self.corpus)

    def generateWhenQuestions(self):
        """
        create when questions
        
        Returns: list of when questions
        """

        for sent in self.spacyCorpus.sents:
            # print(sent)
            # print([(ent, ent.label_) for ent  in sent.ents])
            foundDate, foundEvent = False, False
            whenQuestions = []
            for ent in sent.ents:
                currQuestion, obj = ["When", "[auxVerb]"], None
                if ent.label_ in set(["DATE"]):
                    print("Sentence: {}".format(sent))
                    print("Entity:", ent, ent.label_)
                    foundDate = True
            if foundDate:
                # print([(tok, tok.pos, tok.dep_) for tok in sent])
                for chunk in sent.noun_chunks:
                    if chunk.root.dep_  == "nsubj":
                        currQuestion.append((chunk.text, chunk.root.dep_))
                        currQuestion.append((chunk.root.head.text))

                        # currQuestion.append((chunk.root.head.text))
                    if chunk.root.dep_ == "dobj":
                        obj = (chunk.text, "dobj")
                if obj is not  None:
                    currQuestion.append(obj)
                print(currQuestion)
            print("\n")

    def generateQuestions(self):
        pass

    def chooseNQuestions(self):
        pass


if __name__ == "__main__":
    article, nquestions = sys.argv[1], sys.argv[2]

    article = open(article, "r", encoding="UTF-8").read()
    nquestions = int(nquestions)

    ask = Ask(article, nquestions)
    ask.preprocess()
    ask.generateWhenQuestions()
