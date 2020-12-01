'''
Class that will return a set of answers based on given question text
Command: ./answer article.txt questions.txt
Python command: python answer.py article.txt questions.txt
'''

import re
import sys
import os

import numpy as np
import time
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


        # # read in BERT model and tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "deepset/bert-base-cased-squad2")
        # self.model = AutoModelForQuestionAnswering.from_pretrained(
        #     "deepset/bert-base-cased-squad2")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
        self.qWords = set([WHAT, WHEN, WHO, WHERE, WHY, HOW, WHICH, "WHOSE", "WHOM"])
        self.binWords = set(["IS", "AM", "ARE", "WAS", "WERE", "BE", "BEING", "BEEN", "CAN", "COULD", "DO", "DOES", "DID", "HAS", "HAVE", "HAD", "HAVING", "MAY", "MIGHT", "MUST", "OUGHT", "SHALL", "SHOULD", "WILL", "WOULD"])

    def preprocess(self):
        """[preprocess the corpus and create spacy objects for corpus and questions]
        """

        p = Parse()

        self.corpus = p.parseCorpus(self.article)

        self.questions = re.sub(r"[\n]+", " ", self.questions)
        # print(self.questions)
        self.spacyCorpus = self.nlp(self.corpus)
        # self.spacyQuestions = self.nlp(self.questions)

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
            question = question.strip()
            # Initialize Question Class Object, and start storing information
            parsedQ, newQuestion = [], Question()
            # adding it back, since we split on it
            newQuestion.raw_question = question + "?"
            # Create the spacy doc on this single question
            newQuestion.spacyDoc = self.nlp(question)
            root_i = 0
            rem_word = ""
            question_tok = [token.text for token in newQuestion.spacyDoc]
            #first word is wh question word
            if question_tok[0].upper() in self.qWords:
                newQuestion.question_type = question_tok[0].upper()
                rem_word = question_tok[0]
            reverse_q = False
            #subordinating conjunction / binary
            if newQuestion.question_type in set([WHEN, WHERE, None]):
                punct_found = False
                for token_i in range(len(newQuestion.spacyDoc)):
                    token = newQuestion.spacyDoc[token_i]
                    if token.dep_ == "punct":
                        punct_found = True
                    
                    if token.dep_ == "ROOT":
                        root_i = token_i
                        #reclassify for punct
                        
                        #wh question
                        if punct_found:
                            newQuestion.question_type = None
                            for child in token.children:
                                if child.text.upper() in self.qWords:
                                    newQuestion.question_type = child.text.upper()
                                    rem_word = child.text
                                    break
                        #binary
                        elif root_i > 0 and newQuestion.spacyDoc[root_i - 1].dep_ == "punct":
                            rem_word = ""
                            newQuestion.question_type = BINARY
                            reverse_q = True
                        break
            if newQuestion.question_type == None:
                rem_word = ""
                newQuestion.question_type = BINARY

                 
            # reverse the question for subordinating conjuction cases
            if reverse_q:
                    parsedQ = question_tok[root_i + 1:] + question_tok[:root_i - 1]
            else: #if newQuestion.question_type.upper() in self.qWords or binary but don't reverse it
                for word in question_tok:
                    if word == rem_word:
                        continue
                    parsedQ.append(word)
                
                
            newQuestion.parsed_version = " ".join(parsedQ)
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

    def similarity(self, distFunc=cosine, k=3, model=None):
        """
        Runs the input distance function on all of the questions and compares with the corpus.
        Add in the corpus spacy objects into the .anwsers attribute
        Args:
            distFunc [function]: func for the similarity, defaults to cosine. 
            k [int]: top k answer sentences
        Returns:
            [Question Objects]: list of the question objects
        """
        #TODO: need to check if the k is below the length of the wikipedia corpus lol

        if model is None:
            qs = self.questionProcessing(self.qWords, model="distilbert-base-nli-stsb-mean-token")
            # Run corpus parsing, with the spacy doc object. Return a 2D numpy array, (numSents, len(sentVec))
            cs = self.corpusVector(self.spacyCorpus, model="distilbert-base-nli-stsb-mean-token")
        else:
            qs = self.questionProcessing(self.qWords, model=model)
            cs = self.corpusVector(self.spacyCorpus, model=model)
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
        """
        BERT Function by Eric Liang to extract appropriate answer from a sentence
        """
        # encode and get best possible answer from sentence
        inputs = self.tokenizer.encode_plus(
            str(orgQuestion), str(orgAnswer), return_tensors="pt")
        answer_start_scores, answer_end_scores = self.model(**inputs)
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        correct_tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end])
        return self.tokenizer.convert_tokens_to_string(correct_tokens)
    


    def answerBin(self, answerSent, simScore, qobj):
        """
        Specific workflow for answering binary questions. Prints {yes, no, no answer}
        Finds if the question has negation and negation in the answer to attempt to answer yes or nos
        Args:
            answerSent: str: the phrase that BERT returns to be the "answer"
            simScore: float: the cosine similarity of the original answer and the question
            qObj: question object: the object that holds the question and answerss
        return:
            None
        """
        questionNeg = 0  # 0 question is not negated, 1 question is negated
        # Looking for a neg token in the question
        for t in qobj.spacyDoc:
            # debugPrint(t.text)
            if t.dep_ == "neg":
                questionNeg = 1

        # If the similarity is really low we should just drop
        if simScore < 0.35:  
            print("Answer Not Found")
            return

        # Parsing the answer, for negations of the root verb
        ansDoc, ans = self.nlp(answerSent), None
        for token in ansDoc:
            if token.dep_ == "neg" or token.text == "not" or token.text == "no":
                ans = 0
            if token.dep_ == "ROOT":
                for child in token.children:
                    if child.dep_ == "neg":
                        ans = 0
                        break
                if ans == 0:
                    break
            if ans is None:
                ans = 1

        # print answer as is if question is not negated. If question is negated invert answer
        if questionNeg:
            if ans:
                print("No")
            else:
                print("Yes")
        else:
            if ans:
                print("Yes")
            else:
                print("No")


def ensembleModel(qObjListA, qObjListB):
    """
    Ensemble idea to pick top-k from the model that has a higher cosine score. 
    Args:
        qObjListA: list of question Object: objects created with model distilroberta-base-msmarco-v2
        qObjListB: list of question Object: objects created with model roberta-large-nli-stsb-mean-tokens
    
    returns:
        None
    """
    qIdx = 0
    qsObjLst_marco, qsObjLst_roberta = qObjListA, qObjListB
    for obj_idx in range(len(qsObjLst_marco)):
        # Get question
        qObj = qsObjLst_marco[obj_idx]
        orgQuestion = qObj.raw_question

        debugPrint("Question {}: {}".format(qIdx, qObj.raw_question))
        debugPrint(qObj.question_type)
        for i in range(len(qObj.answers)):

            qObj = qsObjLst_marco[obj_idx]
            marcoScore = qObj.score[i]
            orgAnswer_marco = qObj.answers[i]

            qObj = qsObjLst_roberta[obj_idx]
            robertaScore = qObj.score[i]
            orgAnswer_roberta = qObj.answers[i]

            if robertaScore and marcoScore < 0.31:
                debugPrint("Answers below the cutoff")
                # debugPrint(f"Marco Score: {marcoScore}, Roberta Score: {robertaScore}")
                debugPrint("Marco Answer")
                debugPrint("Answer {}: {} \nCOS SCORE: {}".format(i, orgAnswer_marco,  marcoScore))
                debugPrint("Roberta Answer:\nAnswer {}: {} \nCOS SCORE: {}".format(i, orgAnswer_roberta, robertaScore))

                print("Answer not found!")
                break

            # debugPrint(f"{i} \t Marco: Score:{marcoScore}\n Answer: {orgAnswer_marco}")
            # debugPrint(f"{i} \t Roberta: Score:{robertaScore}\n Answer: {orgAnswer_roberta}")

            if robertaScore < marcoScore:
                debugPrint("Marco was chosen")
                debugPrint("Answer {}: {} \nCOS SCORE: {}".format(i, orgAnswer_marco,  marcoScore))

                foundAnswer = answer.answerQuestion(orgQuestion, orgAnswer_marco)
                if foundAnswer != "[CLS]" and foundAnswer.strip() != "":
                    if qObj.question_type == "BINARY":
                        debugPrint(f"Bert Answer (Input to Binary): {foundAnswer}")
                        answer.answerBin(foundAnswer, qObj.score[i], qObj)
                        break  # We break since we have answered this question
                    debugPrint("BERT ANSWER", end=": ")

                    # Capitalize first letter in first word
                    if len(foundAnswer) != 0:
                        foundAnswer = foundAnswer[0].upper() + foundAnswer[1:]

                    print(foundAnswer)
                    break
                elif i == len(qObj.answers)-1:
                    debugPrint("BERT ANSWER", end=": ")
                    print("Answer not found!")
            else:
                debugPrint("Roberta was chosen")
                debugPrint("Answer {}: {} \nCOS SCORE: {}".format(i, orgAnswer_roberta, robertaScore))

                foundAnswer = answer.answerQuestion(orgQuestion, orgAnswer_roberta)
                if foundAnswer != "[CLS]" and foundAnswer.strip() != "":
                    if qObj.question_type == "BINARY":
                        debugPrint(f"Bert Answer (Input to Binary): {foundAnswer}")
                        answer.answerBin(foundAnswer, qObj.score[i], qObj)
                        break  # We break since we have answered this question
                    debugPrint("BERT ANSWER", end=": ")

                    # Capitalize first letter in first word
                    if len(foundAnswer) != 0:
                        foundAnswer = foundAnswer[0].upper() + foundAnswer[1:]

                    print(foundAnswer)
                    break
                elif i == len(qObj.answers)-1:
                    debugPrint("BERT ANSWER", end=": ")
                    print("Answer not found!")

            debugPrint("\n")

        debugPrint("\n")
        qIdx += 1
    e = time.time()
    debugPrint(f"Answering took {e-s} Seconds")


if __name__ == "__main__":
    s = time.time()
    article, questions = sys.argv[1], sys.argv[2]

    article = open(article, "r", encoding="UTF-8").read()
    questions = open(questions, "r", encoding="UTF-8").read()

    answer = Answer(article, questions)
    answer.preprocess()

    # Run the processing to return back a list of question objects
    # roberta-base-nli-stsb-mean-tokens pretrain semantic textual similarity model
    # distilbert-base-nli-stsb-mean-token also pretrain STS
    # distilroberta-base-msmarco-v2 pretrain for information retrival and 
    # roberta-large-nli-stsb-mean-tokens is the larger version of roberta

    qsObjLst_marco = answer.similarity(model="distilroberta-base-msmarco-v2")
    qsObjLst_roberta = answer.similarity(model="roberta-large-nli-stsb-mean-tokens")
    

    ensembleModel(qsObjLst_marco, qsObjLst_roberta)

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
