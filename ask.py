'''
Class that will return a set of questions based on given text
Command: ./ask article.txt nquestions
Python command: python answer.py article.txt 21
'''

import sys

import numpy as np
import spacy

from parse import Parse

WHAT = "WHAT"
WHEN = "WHEN"
WHERE = "WHERE"
WHO = "WHO"
BINARY = "BINARY"

class Ask:
    def __init__(self, article, nquestions):
        self.article = article
        self.nquestions = nquestions
        self.nlp = spacy.load("en_core_web_md")  # spacy model

        self.questionsGenerated = {
            WHAT: [],
            WHEN: [],
            WHERE: [],
            WHO: [],
            BINARY: [],
        }

        self.keyWords = {
            WHERE: set(["to", "at", "between", "near", "into", "on", "across"])
        }

    def preprocess(self):
        p = Parse()

        self.corpus = p.parseCorpus(self.article)
        self.spacyCorpus = self.nlp(self.corpus)

    def generateWhen(self, sent):
        """
        create when questions
        
        Returns: list of when questions
        """

        # for sent in self.spacyCorpus.sents:
            # print(sent)
            # print([(ent, ent.label_) for ent  in sent.ents])
        foundDate, foundEvent = False, False
        whenQuestions = []
        for ent in sent.ents:
            currQuestion, obj = ["When", "[auxVerb]"], None
            if ent.label_ in set(["DATE"]):
                print("Sentence: {}".format(sent))
                print("Entity: {} {}".format(ent, ent.label_))
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

    def generateBinary(self, sent):
        return None

    def generateWhat(self, sent):
        return None

    def generateWhere(self, sent):
        listOfPrepositions = []
        for tok in sent:
            if tok.pos_ == "ADP":
                if tok.text not in self.keyWords[WHERE]:
                    continue
                prepositionalPhrase = ' '.join([t.orth_ for t in tok.subtree])
                listOfPrepositions.append(prepositionalPhrase)
        
        # go through each one in pps
        # generate question

    def generateWho(self, sent):
        return None
         
    def generateQuestions(self):
        for sent in self.spacyCorpus.sents:
            # generate binary question
            # generate what question
            # generate where question
            print(self.generateWhere(sent))
            # generate who question
            # generate when question

            # either add all questions we generate to the dict
            # OR pick one question and add to a result list
            



    def chooseNQuestions(self):
        pass


if __name__ == "__main__":
    article, nquestions = sys.argv[1], sys.argv[2]

    article = open(article, "r", encoding="UTF-8").read()
    nquestions = int(nquestions)

    ask = Ask(article, nquestions)
    ask.preprocess()
    ask.generateQuestions()
    # ask.generateWhenQuestions()
