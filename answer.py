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
from scipy.spatial import distance
from distUtils import jaccardSim, cosine, diceSim
from question import  Question


def scipyJaccard(u, v):
    return 1 - distance.jaccard(u, v)

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

    def questionProcessing(self, excludeTokens=None):
        """[get the sentence vector for a given doc]

        Args:
            doc ([spacy]): [spacy model from text]
            excludeTokens ([set], optional): [tokens to exclude]. Defaults to None.

        Returns:
            [list]: list of Numpy Arrays of Sentence vector
        """
        # This is the only model I tried. First time running should cause a download but afterwards it doesnt download.
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')

        # if excludeTokens is None:
        #     sentence_embeddings = model.encode(sentences) 
        #     return sentence_embeddings
        # Gonna build the question without question words and '?'
        parsedQuestions = []
        for question in self.questions.split("?"):
            # print(sentence)
            parsedWords = []
            newQuestion = Question()
            newQuestion.raw_question = question + "?"
            newQuestion.spacyDoc = self.nlp(question)
            for word in question.split(" "):
                # if given excludeTokens, skip word if it's in excludeTokens
                if word.upper() in excludeTokens:
                    # if word == "?":
                    #     continue
                    newQuestion.question_type = word.upper()
                    continue
                # question.question_type = "Other"
                # I dont want to remove stopping since we are looking at the sentence level now
                parsedWords.append(word)
            
            newQuestion.parsed_version  = " ".join(parsedWords)
            print(newQuestion.parsed_version)
            # going to store this in  a list just  in case it wants 2d inputs only
            newQuestion.sent_vector = model.encode(newQuestion.parsed_version)  
            
            parsedQuestions.append(newQuestion) # Now we join all of the word back together 

        # Takes in a list of strings, careful to feed in string and not spacy objects
        # Returns a list of numpy arrays
        # sentence_embeddings = model.encode(sentences) 
        # assert(len(sentence_embeddings) == len(list(doc.sents)))
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


    def similarity(self, distFunc=cosine):

        excludeTokens = set(["WHO", "WHAT", "WHEN", "WHERE", "HOW", "?"])

        qs = self.questionProcessing(excludeTokens)
        cs = self.corpusVector(self.spacyCorpus)

        for i in range(len(qs)):
            dists = np.apply_along_axis(distFunc, 1, cs, qs[i].sent_vector)

            print("Question:", qs[i].raw_question)

            #print("Answer:", list(self.spacyCorpus.sents)[np.argmax(cos)])
            ind = dists.argsort()[-3:][::-1] # we might want to look at numbers later?
            for j in range(3):
                print("Answer", j, ":", list(self.spacyCorpus.sents)[ind[j]])
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
    answer.similarity(distFunc  = jaccardSim)
    # a = [1.5, 3.45, 5, 0, 23]
    # b = [342, 1, 3, 1000, 3.9]
    # c = [1.5, 3.45, 5, 0, 23]
    # print(1 - scipyJaccard(a, b))
    # print(jaccardSim(a, b))

    # print(1 - scipyJaccard(a, c))
    # print(jaccardSim(a, c))