'''
Class that will return a set of answers based on given question text
Command: ./answer article.txt questions.txt
Python command: python answer.py article.txt questions.txt
'''

import re
import sys
import os

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer  # Pip installed
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from distUtils import cosine, diceSim, jaccardSim, scipyJaccard
from parse import Parse
from question import Question

# If false the log function will not print
DEBUG = True

def debugPrint(s, **kwargs):
    if DEBUG: print(s, **kwargs)

# Initialization of question words
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

        self.nlp = spacy.load("en_core_web_lg")  # spacy model

        # read in BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepset/bert-base-cased-squad2")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            "deepset/bert-base-cased-squad2")

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

    def questionProcessing(self, qWords=None, model=None):
        """ Specialized parsing for the questions. 
        Args:
            qWords ([set]): [question words, "WHO", "WHAT", ...]
        Returns:
            [list]: list of Question Object, each stores info on the question: type, vec, raw, parsed
        """
        # This is the only model I tried. First time running should cause a download but afterwards it doesnt download.
        # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        if model is None:
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        else:
            model = SentenceTransformer(model)
        # model = SentenceTransformer('distilroberta-base-msmarco-v2')
        parsedQuestions = []

        # Since we are only looking at questions, we can split on '?'
        for question in self.questions.split("?"):
            # If the sentence is all whitespace go next, mostly for blank end lines
            if question.isspace() or len(question) == 0:
                continue
            # Initialize Question Class Object, and start storing information
            parsedQ, newQuestion = [], Question()
            # adding it back, since we split on it
            newQuestion.raw_question = question + "?"
            # Create the spacy doc on this single question
            newQuestion.spacyDoc = self.nlp(question)

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
            # Now we join all of the word back together
            newQuestion.parsed_version = " ".join(parsedQ)
            # print(newQuestion.parsed_version)

            newQuestion.sent_vector = model.encode(newQuestion.parsed_version)
            parsedQuestions.append(newQuestion)

        return parsedQuestions

    def corpusVector(self, doc, excludeTokens=None, model=None):
        """[get the sentence vector for a given doc]
        Args:
            doc ([spacy]): [spacy model from text]
            excludeTokens ([set], optional): [tokens to exclude]. Defaults to None.
        Returns:
            [list]: list of Numpy Arrays of Sentence vector
        """
        # This is the only model I tried. First time running should cause a download but afterwards it doesnt download.
        if model is None:
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        else:
            model = SentenceTransformer(model)
        # model = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")
        # model = SentenceTransformer('distilroberta-base-msmarco-v2')


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

    def similarity(self, distFunc=cosine, k=3):
        """
        Runs the input distance function on all of the questions and compares with the corpus.
        Add in the corpus spacy objects into the .anwsers attribute
        Args:
            distFunc [function]: func for the similarity, defaults to cosine. 
            k [int]: top k answer sentences
        Returns:
            [Question Objects]: list of the question objects
        """

        qWords = set([WHAT, WHEN, WHO, WHERE, WHY, HOW, WHICH])

        # Run the processing to return back a list of question objects
        # roberta-base-nli-stsb-mean-tokens pretrain semantic textual similarity model
        # distilbert-base-nli-stsb-mean-token also pretrain STS
        # distilroberta-base-msmarco-v2 pretrain for information retrival and 

        qs = self.questionProcessing(qWords, model="distilroberta-base-msmarco-v2")

        # Run corpus parsing, with the spacy doc object. Return a 2D numpy array, (numSents, len(sentVec))
        cs = self.corpusVector(self.spacyCorpus, model="distilroberta-base-msmarco-v2")

        # For every question
        for i in range(len(qs)):

            # Apply the dist similarity function down the numpy array of the corpuss
            dists = np.apply_along_axis(distFunc, 1, cs, qs[i].sent_vector)

            # sorting for top k
            # test different k values (hyperparameter)
            ind = dists.argsort()[-k:][::-1]

            for j in range(k):
                spacyCorpusList = list(self.spacyCorpus.sents)

                # Add this answer to question object
                qs[i].answers.append(spacyCorpusList[ind[j]])
                qs[i].score.append(dists[ind[j]])
        return qs

    def answerQuestion(self, orgQuestion, orgAnswer):
        # encode and get best possible answer from sentence
        inputs = self.tokenizer.encode_plus(
            str(orgQuestion), str(orgAnswer), return_tensors="pt")
        answer_start_scores, answer_end_scores = self.model(**inputs)
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        correct_tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end])
        return self.tokenizer.convert_tokens_to_string(correct_tokens)


if __name__ == "__main__":
    article, questions = sys.argv[1], sys.argv[2]

    article = open(article, "r", encoding="UTF-8").read()
    questions = open(questions, "r", encoding="UTF-8").read()

    answer = Answer(article, questions)
    answer.preprocess()
    qsObjLst = answer.similarity()

    # print statements used to debug preprocessing and similarity matching
    """
    for qObj in qsObjLst:
        print("Question: {} TYPE: {}".format(qObj.raw_question, qObj.question_type))
        print("Named Entity: {}".format([ent for ent in qObj.spacyDoc.ents]), "\n")
        for i in range(len(qObj.answers)):
            print("Answer {}: {}".format(i, qObj.answers[i]))
            print("Named Entities: {}".format([ent for ent in qObj.answers[i].ents]), "\n")
        print("\n")
    """
    qIdx = 0
    for qObj in qsObjLst:
        # Get question
        orgQuestion = qObj.raw_question
        debugPrint("Question {}: {}".format(qIdx, qObj.raw_question))

        for i in range(len(qObj.answers)):
            # Get answer
            orgAnswer = qObj.answers[i]
            debugPrint("Answer {}: {} \nCOS SCORE: {}".format(i, orgAnswer, qObj.score[i]))

            # debugPrint("Found Answer:")
            foundAnswer = answer.answerQuestion(orgQuestion, orgAnswer)
            if foundAnswer != "[CLS]" and foundAnswer.strip() != "":
                debugPrint("BERT ANSWER", end=": ")
                print(foundAnswer)
                break
            elif i == len(qObj.answers)-1:
                debugPrint("BERT ANSWER", end=": ")
                print("Answer not found!")
            debugPrint("\n")

        debugPrint("\n")
        qIdx += 1

    # Used for manual unit testing of a case
    """
    orgAnswer = "Pittsburgh was named in 1758 by General John Forbes, in honor of British statesman William Pitt, 1st Earl of Chatham."
    orgQuestion = "When was Pittsburgh named by General John Forbes, in honor of British statesman William Pitt, 1st Earl of Chatham?"
    print(answer.answerQuestion(orgQuestion, orgAnswer))
    orgQuestion = "What was named in 1758 by General John Forbes?"
    print(answer.answerQuestion(orgQuestion, orgAnswer))
    orgQuestion = "Who named Pittsburgh in 1758?"
    print(answer.answerQuestion(orgQuestion, orgAnswer))
    orgQuestion = "In honor of whom was Pittsburgh named in 1758 by General John Forbes?"
    print(answer.answerQuestion(orgQuestion, orgAnswer))

    orgAnswer = "The first is called the Meidum pyramid, named for its location in Egypt first."
    orgQuestion = "Who was the first Pharaoh of the Old Kingdom?"
    print(answer.answerQuestion(orgQuestion, orgAnswer))
    """

    # a = [1.5, 3.45, 5, 0, 23]
    # b = [342, 1, 3, 1000, 3.9]
    # c = [1.5, 3.45, 5, 0, 23]
    # print(1 - scipyJaccard(a, b))
    # print(jaccardSim(a, b))

    # print(1 - scipyJaccard(a, c))
    # print(jaccardSim(a, c))
