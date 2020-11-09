'''
Class that will return a set of answers based on given question text
Command: ./answer article.txt questions.txt
Python command: python answer.py article.txt questions.txt
'''

from parse import Parse
from question import QuestionProcess
from sentence_transformers import SentenceTransformer  # Pip installed
import sys
import spacy
import numpy as np
import re
from scipy.spatial import distance


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Should be vectorized eventually. this is kinda a zzz
# using 1 - jaccardSim bc it's weird like that (:


def jaccardSim(u, v):
    minSum, maxSum = 0, 0
    for i in range(len(u)):
        minSum += min([u[i], v[i]])
        maxSum += max([u[i], v[i]])
    return 1 - minSum/maxSum


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
            # This value is hardcoded from the Spacy Word2Vec model
            accum = np.zeros((300,))
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

            # Now we join all of the word back together
            sentences.append(" ".join(parsedSentence))

        # Takes in a list of strings, careful to feed in string and not spacy objects
        # Returns a list of numpy arrays
        sentence_embeddings = model.encode(sentences)
        # assert(len(sentence_embeddings) == len(list(doc.sents)))
        return sentence_embeddings

    def similarity(self, distFunc=cosine):

        excludeTokens = set(["WHO", "WHAT", "WHEN", "WHERE", "HOW", "?"])

        qs = self.getSentenceVector(self.spacyQuestions, excludeTokens)
        cs = self.getSentenceVector(self.spacyCorpus)

        for i in range(len(qs)):
            dists = np.apply_along_axis(distFunc, 1, cs, qs[i])

            print("Question:", list(self.spacyQuestions.sents)[i])

            #print("Answer:", list(self.spacyCorpus.sents)[np.argmax(cos)])
            # we might want to look at numbers later?
            ind = dists.argsort()[-3:][::-1]
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
    answer.similarity(distFunc=jaccardSim)
    # a = [1.5, 3.45, 5, 0, 23]
    # b = [342, 1, 3, 1000, 3.9]
    # c = [1.5, 3.45, 5, 0, 23]
    # print(1 - scipyJaccard(a, b))
    # print(jaccardSim(a, b))

    # print(1 - scipyJaccard(a, c))
    # print(jaccardSim(a, c))
