'''
Class that will return a set of answers based on given question text
Command: ./answer article.txt questions.txt
Python command: python answer.py article.txt questions.txt
'''

import re
import sys

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer  # Pip installed

from distUtils import cosine, diceSim, jaccardSim, scipyJaccard
from parse import Parse
from question import Question

WHAT = "WHAT"
WHEN = "WHEN"
WHO = "WHO"
WHERE = "WHERE"
WHY = "WHY"
HOW = "HOW"
WHICH = "WHICH"
BINARY = "BINARY"



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
        # print(self.questions)
        self.spacyCorpus = self.nlp(self.corpus)
        # self.spacyQuestions = self.nlp(self.questions)

    # depreciated
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

    def questionProcessing(self, qWords=None):
        """ Specialized parsing for the questions. 

        Args:
            qWords ([set]): [question words, "WHO", "WHAT", ...]

        Returns:
            [list]: list of Question Object, each stores info on the question: type, vec, raw, parsed
        """
        # This is the only model I tried. First time running should cause a download but afterwards it doesnt download.
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        parsedQuestions = []

        # Since we are only looking at questions, we can split on '?'
        for question in self.questions.split("?"):
            # If the sentence is all whitespace go next, mostly for blank end lines
            if question.isspace():
                continue
            # Initialize Question Class Object, and start storing information
            parsedQ, newQuestion = [], Question()
            newQuestion.raw_question = question + "?"  # adding it back, since we split on it
            newQuestion.spacyDoc = self.nlp(question)  # Create the spacy doc on this single question

            # Remove the question word, categorize the question, and get its vector with sentence Transformer
            for word in question.split(" "):

                # If word in qWords, we have found the question class, and dont add to parsedQ
                # This does not solve the issues with 'can you repeat what elmo said?' 
                if word.upper() in qWords:
                    newQuestion.question_type = word.upper()
                    continue
                parsedQ.append(word)
            
            # If the question_type was not set, it means lacks a question word, therefore should be Binary/other
            if newQuestion.question_type is None:
                newQuestion.question_type = BINARY
            newQuestion.parsed_version  = " ".join(parsedQ)  # Now we join all of the word back together 
            # print(newQuestion.parsed_version)

            newQuestion.sent_vector = model.encode(newQuestion.parsed_version)  
            parsedQuestions.append(newQuestion) 

        return parsedQuestions

    def corpusVector(self, doc, excludeTokens=None):
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


    def similarity(self, distFunc=cosine, k=3):
        """
        Runs the input distance function on all of the questions and compares with the corpus.
        Prints out the top k sentence matches
        Args:
            distFunc [function]: func for the similarity, defaults to cosine. 
            k [int]: top k answer sentences to print with the question
        Returns:
            None
        """

        qWords = set([WHAT, WHEN, WHO, WHERE, WHY, HOW, WHICH])

        qs = self.questionProcessing(qWords)
        cs = self.corpusVector(self.spacyCorpus)

        for i in range(len(qs)):
            dists = np.apply_along_axis(distFunc, 1, cs, qs[i].sent_vector)

            print("Question:", qs[i].raw_question, "Type: {}".format(qs[i].question_type))

            #print("Answer:", list(self.spacyCorpus.sents)[np.argmax(cos)])
            ind = dists.argsort()[-k:][::-1] # we might want to look at numbers later?
            for j in range(k):
                print("Answer", j, ":", list(self.spacyCorpus.sents)[ind[j]])
            print("\n")

    def answerQuestion(self, question, sentence):
        pass


if __name__ == "__main__":
    article, questions = sys.argv[1], sys.argv[2]

    article = open(article, "r").read()
    questions = open(questions, "r").read()

    answer = Answer(article, questions)
    answer.preprocess()
    answer.similarity(distFunc  = jaccardSim)
    # a = [1.5, 3.45, 5, 0, 23]
    # b = [342, 1, 3, 1000, 3.9]
    # c = [1.5, 3.45, 5, 0, 23]
    # print(1 - scipyJaccard(a, b))
    # print(jaccardSim(a, b))

    # print(1 - scipyJaccard(a, c))
    # print(jaccardSim(a, c))
