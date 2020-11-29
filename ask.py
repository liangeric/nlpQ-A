'''
Class that will return a set of questions based on given text
Command: ./ask article.txt nquestions
Python command: python answer.py article.txt 21
'''

import sys
import random

import numpy as np
import spacy
import time 

from parse import Parse

def debugPrint(s, **kwargs):
    if DEBUG: print(s, **kwargs)

WHAT = "What"
WHEN = "When"
WHERE = "Where"
WHO = "Who"
BINARY = "Binary"


class Ask:
    def __init__(self, article, nquestions):
        """
        Parameters
        ----------
        article : str
            File path to the article to be used as the corpus
        nquestions : int
            The number of questions to generate
        """
        self.article = article
        self.nquestions = nquestions
        self.nlp = spacy.load("en_core_web_md")  # spacy model

        self.questionsGenerated = {
            WHAT: set(),
            WHEN: set(),
            WHERE: set(),
            WHO: set(),
            BINARY: set(),
        }

        self.keyWords = {
            WHERE: set(["to", "at", "between", "near", "into", "on", "across"])
        }

    def preprocess(self):
        """Method that will preprocess the corpus and intialize the spacy model

        Parameters
        ----------
        Returns
        -------
        """
        p = Parse()

        self.corpus = p.parseCorpus(self.article)
        self.spacyCorpus = self.nlp(self.corpus)

    def generateWhen(self, sent):
        """
        create when questions

        Returns: list of when questions
        """

        for sent in self.spacyCorpus.sents:
            foundDate, foundEvent = False, False
            whenQuestions = []
            for ent in sent.ents:
                currQuestion, obj = ["When", "[auxVerb]"], None
                if ent.label_ in set(["DATE"]):
                    foundDate = True
            if foundDate:
                plural = False  # default to singular
                for chunk in sent.noun_chunks:
                    if chunk.root.dep_ == "nsubj":
                        currQuestion.append(chunk.text)
                        assert(type(chunk.text) == str)
                        currQuestion.append(chunk.root.head.lemma_)
                        assert(type(chunk.root.head.lemma_) == str)
                        rootVerb = chunk.root.head
                        # plural vs singular
                        if chunk.root.tag_ == "NNPS" or chunk.root.tag_ == "NNS":
                            plural = True
                        tagMap = self.nlp.vocab.morphology.tag_map[chunk.root.head.tag_]
                        pastTense = False  # default to present tense
                        if "Tense_past" in tagMap and tagMap["Tense_past"] == True:
                            pastTense = True

                    if chunk.root.dep_ == "dobj" and chunk.root.head.text == rootVerb.text:
                        obj = chunk.text

                if obj is not None:
                    currQuestion.append(obj)
                    assert(type(obj) == str)
                #  re parse for preposition and the object that relates to it.
                # for word in sent:
                #     prep = None
                #     if word.dep_ == "prep" and word.head.text == rootVerb:
                #         currQuestion.append((word, "prep"))
                #         prep = word.text
                #     if prep is not  None and word.head.text == prep:
                #         currQuestion.append((word, "pobj"))
                if rootVerb.lemma_ == "be":
                    currQuestion = currQuestion[0:-1]
                    if "[auxVerb]" in currQuestion:
                        currQuestion.remove("[auxVerb]")
                    currQuestion.insert(1, rootVerb.text)
                    assert(type(rootVerb.text) == str)
                else:
                    if pastTense:
                        conjugatedVerb = "did"
                    else:  # presentTense
                        if plural:
                            conjugatedVerb = "do"
                        else:  # singular
                            conjugatedVerb = "did"

                    if "[auxVerb]" in currQuestion:
                        currQuestion.remove("[auxVerb]")
                    currQuestion.insert(1, conjugatedVerb)
                if len(currQuestion) > 2:
                    q = " ".join(currQuestion[1:])
                    self.addQuestionToDict(q, WHEN)
            

    def generateBinary(self, sent):
        """Method that will look generate a binary question from ROOT AUX
        TODO: Needs work on handling complex questions and weird edge cases

        Parameters
        ----------
        sent: spacy span
            A sentence from the corpus
        Returns
        -------
        """
        question = ""
        for token in sent:
            # print(
            #     f"{token.text:<20}{token.pos_:<20}{token.dep_:<20}{token.head.text:<20}")
            if token.pos_ == "AUX" and token.dep_ == "ROOT":
                question_word = token.text.capitalize()
                question_body = ''.join(
                    t.text_with_ws.lower() for t in self.spacyCorpus[sent.start:sent.end-1] if t.i != token.i)
                question = f"{question_word} {question_body}"
                break

        self.addQuestionToDict(question, BINARY)

    def generateWhAux(self, sent):
        """Method that will look for AUX pos and will generate WHO, WHAT, WHERE questions accordingly

        Parameters
        ----------
        sent: spacy span
            A sentence from the corpus
        Returns
        -------
        """
        WHAT_ENT = set(["ORG", "PRODUCT"])
        WHERE_ENT = set(["LOC", "GPE"])
        WHO_ENT = set(["PERSON"])
        for chunk in sent.noun_chunks:
            head = chunk.root.head

            if head.pos_ == "AUX":

                question_type = None
                for ent in sent.ents:

                    if not ent or not chunk:
                        continue
                    equal = ent.text.lower() == chunk.text.lower()
                    if equal and ent.label_ in WHO_ENT:
                        question_type = WHO
                        break

                    elif equal and ent.label_ in WHAT_ENT:
                        question_type = WHAT
                        break

                    elif equal and ent.label_ in WHERE_ENT:
                        question_type = WHERE
                        break

                if question_type is not None:
                    self.addQuestionToDict(
                        f"{head.text} {chunk.text}", question_type)

    def generateWhere(self, sent):
        """Main method that will generate where questions given a sentence

        Parameters
        ----------
        sent: spacy span
            A sentence from the corpus
        Returns
        -------
        """
        listOfPrepositions = []
        for tok in sent:
            if tok.pos_ == "ADP":
                if tok.text not in self.keyWords[WHERE]:
                    continue
                prepositionalPhrase = ' '.join([t.text for t in tok.subtree])
                listOfPrepositions.append(prepositionalPhrase)

        # go through each one in pps
        # generate question
        # TODO: Actually generate where questions based on preposition, DOES NOT WORK YET
        # print(listOfPrepositions)
        return listOfPrepositions

    def generateWho(self, sent):
        """Main method that will generate who questions given a sentence

        Parameters
        ----------
        sent: spacy span
            A sentence from the corpus
        Returns
        -------
        """
        questions = []
        verbs = set(["AUX", "VERB"])
        ner = set(["PERSON"])
        pos = set(["PRON"])

        for token in sent:
            if token.ent_type_ in ner or token.pos_ in pos:
                head = token.head
                if head.pos_ in verbs and head.dep_ == "ROOT":

                    questions.append(
                        ''.join(t.text_with_ws for t in self.spacyCorpus[head.i:sent.end-1]))

        for q in questions:
            self.addQuestionToDict(q, WHO)

    def generateWhat(self, sent):
        """Main method that will generate what questions given a sentence

        Parameters
        ----------
        sent: spacy span
            A sentence from the corpus
        Returns
        -------
        """
        questions = []
        verbs = set(["AUX", "VERB"])
        ner = set(["ORG", "PRODUCT"])

        for token in sent:
            if token.ent_type_ in ner:
                head = token.head
                if head.pos_ in verbs and head.dep_ == "ROOT":

                    questions.append(
                        ''.join(t.text_with_ws for t in self.spacyCorpus[head.i:sent.end-1]))

        for q in questions:
            self.addQuestionToDict(q, WHAT)

    def addQuestionToDict(self, question, TYPE):
        """Method that adds a particular question to the dict based on question type
        It will also add the question type word in front and ? at the end

        Parameters
        ----------
        question: str
            The question in string format without question type word and question mark
        TYPE : str
            The question type that should be added in the front and is used to lookup the 
            set of questions in self.questionsGenerated to add
        Returns
        -------
        """
        if question is None and TYPE is None:
            return

        if len(question):
            question = question.strip()
            if TYPE == BINARY:
                completed_question = f"{question}?"
            else:
                completed_question = f"{TYPE} {question}?"
            # TODO: check the grammar of this generated question
            # TODO: make sure a question that is essentially the same but not 100% the same as another question is not added

            self.questionsGenerated[TYPE].add(completed_question)


    def generateQuestions(self):
        """Method that handles generating questions for each sentence in our corpus

        Parameters
        ----------
        Returns
        -------
        """
        for sent in self.spacyCorpus.sents:
            self.generateWhat(sent)
            self.generateWho(sent)
            self.generateWhAux(sent)
            self.generateBinary(sent)
            self.generateWhen(sent)
            # self.generateWhere(sent) # this method is not completed yet

    def chooseNQuestions(self):
        """Method that handles last part in the pipeline. Randomly picks question type
        and randomly picks a question from the generated questions to print out.

        Parameters
        ----------
        Returns
        -------
        """
        question_types = list(self.questionsGenerated.keys())
        while self.nquestions > 0:
            noMoreQuestions = False
            for i in range(len(question_types)):
                wh = question_types[i]
                if len(self.questionsGenerated[wh]) > 1:
                    break
                # no questions
                elif i == len(question_types) - 1 and len(self.questionsGenerated[wh]) == 0:
                    noMoreQuestions = True
                    break                    

            if noMoreQuestions:
                #print("Unable to generate more questions")
                break
            else:
                current_question_type = random.choice(question_types)
                current_questions_set = self.questionsGenerated[current_question_type]
                if not len(current_questions_set):
                    self.questionsGenerated.pop(current_question_type, None)
                    question_types.remove(current_question_type)
                else:
                    pick_question = random.sample(current_questions_set, 1)[0]
                    if len(pick_question) > 100:  # to ensure better quality of questions
                        current_questions_set.remove(pick_question)
                        continue
                    print(pick_question)
                    current_questions_set.remove(pick_question)
                    self.nquestions -= 1

    '''
    def printGeneratedQuestions(self, TYPE=None):
        """Utility method that prints out all the questions based on question type for debugging purposes
        Do not use this method in the file program, this is only for debugging purposes.

        Parameters
        ----------
        TYPE : str, optional
            The question type used to lookup questions in self.questionsGenerated
            Prints all the questions if TYPE is None
        Returns
        -------
        """
        print(f"---------- PRINTING {TYPE} QUESTIONS ----------")
        for q_type, questions in self.questionsGenerated.items():
            if TYPE is None or q_type == TYPE:
                for q in questions:
                    if q is not None and q != "":
                        print(q)
    '''


if __name__ == "__main__":
    s = time.time()
    article, nquestions = sys.argv[1], sys.argv[2]

    article = open(article, "r", encoding="UTF-8").read()
    nquestions = int(nquestions)

    ask = Ask(article, nquestions)
    ask.preprocess()

    ask.generateQuestions()
    # ask.printGeneratedQuestions(WHAT)
    # ask.printGeneratedQuestions(WHO)
    # ask.printGeneratedQuestions(WHERE)
    # ask.printGeneratedQuestions(WHEN)
    ask.chooseNQuestions()
    e = time.time()
    debugPrint(f"Question asking took {e-s}")
