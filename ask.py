'''
Class that will return a set of questions based on given text
Command: ./ask article.txt nquestions
Python command: python answer.py article.txt 21
'''

import sys
import random

import numpy as np
import spacy

from parse import Parse

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

                if chunk.root.dep_ == "nsubj":
                    currQuestion.append((chunk.text, chunk.root.dep_))
                    currQuestion.append((chunk.root.head.text))

                    # currQuestion.append((chunk.root.head.text))
                if chunk.root.dep_ == "dobj":
                    obj = (chunk.text, "dobj")
            if obj is not None:
                currQuestion.append(obj)
            print(currQuestion)

    def generateBinary(self, sent):
        return None

    def generateWhAux(self, sent):
        """Method that will look for AUX pos and will generate WHO, WHAT, WHERE questions accordingly

        Parameters
        ----------
        sent: spacy span
            A sentence from the corpus
        Returns
        -------
        """
        for chunk in sent.noun_chunks:
            head = chunk.root.head

            if head.pos_ == "AUX":

                question_type = None
                for ent in sent.ents:

                    if not ent or not chunk:
                        continue
                    equal = ent.text.lower() == chunk.text.lower()

                    if equal and ent.label_ in set(["ORG", "PRODUCT"]):
                        question_type = WHAT
                        break

                    elif equal and ent.label_ in set(["LOC", "GPE"]):
                        question_type = WHERE
                        break

                    elif equal and ent.label_ in set(["PERSON"]):
                        question_type = WHO
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
        print(listOfPrepositions)
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
            if not len(question_types):
                print("Unable to generate more questions 😂")
                break
            else:
                current_question_type = random.choice(question_types)
                current_questions_set = self.questionsGenerated[current_question_type]
                if not len(current_questions_set):
                    self.questionsGenerated.pop(current_question_type, None)
                    question_types.remove(current_question_type)
                else:
                    pick_question = random.sample(current_questions_set, 1)[0]
                    print(pick_question)
                    current_questions_set.remove(pick_question)
                    self.nquestions -= 1

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


if __name__ == "__main__":
    article, nquestions = sys.argv[1], sys.argv[2]

    article = open(article, "r", encoding="UTF-8").read()
    nquestions = int(nquestions)

    ask = Ask(article, nquestions)
    ask.preprocess()
    ask.generateQuestions()
    # ask.printGeneratedQuestions(WHAT)
    # ask.printGeneratedQuestions(WHO)
    # ask.printGeneratedQuestions(WHERE)
    ask.chooseNQuestions()
