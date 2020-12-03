'''
Class that will return a set of questions based on given text
Command: ./ask article.txt nquestions
Python command: python ask.py article.txt 21
'''

import sys
import random
import copy

import numpy as np
import spacy
import time

import time


from fuzzywuzzy import process

from parse import Parse


WHAT = "What"
WHEN = "When"
WHERE = "Where"
WHO = "Who"
BINARY = "Binary"

DEBUG = True


def debugPrint(s, **kwargs):
    if DEBUG:
        print(s, **kwargs)


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

        Parameters
        ----------
        sent: spacy span
            A sentence from the corpus
        Returns
        -------
        """
        question = ""
        for token in sent:
            # self.print_token(token)
            if token.pos_ == "AUX" and token.dep_ in ["ROOT", "ccomp"]:
                for child in token.children:
                    if child.dep_ == "nsubj":
                        child_subtree = list(child.subtree)
                        if len(child_subtree) > 0:
                            first, last = child_subtree[0], child_subtree[-1]
                            subj = ''.join(
                                t.text_with_ws for t in self.spacyCorpus[first.i: last.i + 1])

                            question_word = token.text.capitalize()
                            question_body = subj + \
                                ''.join(
                                    t.text_with_ws for t in self.spacyCorpus[token.i + 1: sent.end-1])

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
        whereKeyWords = set(["to", "at", "between", "near",
                             "into", "on", "across", "in"])

        has_where_ent = False
        for ent in sent.ents:
            if ent.label_ in ["GPE", "LOC"]:
                question = "is " + ent.text
                self.addQuestionToDict(question, WHERE)
                has_where_ent = True

        if not has_where_ent:
            return

        for token in sent:

            if token.dep_ == "prep" and (token.head.pos_ == "AUX" or token.head.pos_ == "VERB"):
                if token.text not in whereKeyWords:
                    continue
                head_token = token.head

                for child in head_token.children:
                    # self.print_token(child)
                    if child.dep_ in ["nsubj", 'nsubjpass']:
                        hasPROPN = False
                        for t in child.subtree:
                            if t.pos_ == "PROPN":
                                hasPROPN = True
                                break

                        if hasPROPN:
                            aux_verb = self.spacyCorpus[head_token.i-1]
                            if aux_verb.pos_ != "AUX":
                                aux_verb = None

                            noun_phrase = ''.join(
                                t.text_with_ws for t in child.subtree)

                            noun_phrase = noun_phrase if not aux_verb else aux_verb.text_with_ws + noun_phrase
                            question = noun_phrase + head_token.text_with_ws
                            # print(question,  head_token.text, head_token.lemma_,
                            #       head_token.pos_, head_token.dep_, token.text, list(token.subtree))
                            self.addQuestionToDict(question, WHERE)

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
        good_pos = set(["anybody", "anyone", "everybody", "everyone", "he", "her", "who"
                "herself", "him", "himself", "I", "me", "no one", "nobody", "she", "she",
                "somebody", "someone", "they", "them", "us", "thou", "we", "you"])

        for token in sent:
            if token.ent_type_ in ner or token.text.lower() in good_pos:
                head = token.head
                if head.pos_ in verbs and head.dep_ == "ROOT":
                    questions.append(''.join(t.text_with_ws for t in self.spacyCorpus[head.i:sent.end-1]))


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
                    questionType = WHAT

                    q = ''.join(t.text_with_ws for t in self.spacyCorpus[head.i:sent.end-1])

                    qLower = q.lower()
                    tempSplit = qLower.split(" ")

                    if "he" in tempSplit or "she" in tempSplit:
                        questionType = WHO

                    self.addQuestionToDict(q,questionType)

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

        if len(question) > 0:
            question = question.strip()
            spacy_question = self.nlp(question)
            question = self.lowercase_non_propernouns(spacy_question, TYPE)

            if TYPE == BINARY:
                completed_question = f"{question}?"
            else:
                completed_question = f"{TYPE} {question}?"

            self.questionsGenerated[TYPE].add(completed_question)

    def lowercase_non_propernouns(self, sent, TYPE):
        """This method will take care of case issues when creating a question. 
        Essentially, will remove cases and keep cases in the beginning of a sentence

        Parameters
        ----------
        sent: spacy
            The spacy question that needs to be modified
        TYPE : str
            The question type used to deal with lowercasing
        Returns
        -------
        """
        question_tokens = []
        for token in sent:
            text = token.text_with_ws

            if TYPE == BINARY and token.i == 1 and token.pos_ != "PROPN":
                text = text.lower()

            if TYPE != BINARY and token.i == 0 and token.pos_ != "PROPN":
                text = text.lower()

            question_tokens.append(text)

        return ''.join(question_tokens)

    def generateQuestions(self):
        """Method that handles generating questions for each sentence in our corpus

        Parameters
        ----------
        Returns
        -------
        """
        number_sents = len(list(self.spacyCorpus.sents))
        sent_start_end = {}
        for index, sent in enumerate(self.spacyCorpus.sents):
            sent_start_end[index] = {
                "start": sent.start,
                "end": sent.end
            }

        number_sentences_seen = 0

        corpus_sents_index = [i for i in range(number_sents)]

        while number_sentences_seen < 200:

            if len(corpus_sents_index) <= 0:
                break

            sent_index = random.choice(corpus_sents_index)

            sent_start = sent_start_end[sent_index]["start"]
            sent_end = sent_start_end[sent_index]["end"]

            sent = self.spacyCorpus[sent_start: sent_end]

            self.generateWhat(sent)
            self.generateWho(sent)
            self.generateWhAux(sent)
            self.generateBinary(sent)
            self.generateWhen(sent)
            self.generateWhere(sent)  # this method is not completed yet

            corpus_sents_index.remove(sent_index)
            number_sentences_seen += 1

    def score_questions(self):
        """Method that will score questions and sort them in a list of dict

        Parameters
        ----------
        Returns
        -------
        generatedQuestions : dict
            Dictionary of question types and questions
        """
        generatedQuestions = {}
        for q_type, questions in self.questionsGenerated.items():
            scored_questions = {}
            for q in questions:
                current_score = 0

                if len(q) < 200 and len(q) > 100:
                    current_score += 12
                elif len(q) <= 100 and len(q) > 50:
                    current_score += 10
                elif len(q) <= 50:
                    current_score += 8
                else:
                    current_score += 5

                scored_questions[q] = current_score

            sorted_by_score = sorted(
                scored_questions.items(), key=lambda x: x[1])

            final_sorted_questions = [q[0] for q in sorted_by_score]
            generatedQuestions[q_type] = final_sorted_questions

        return generatedQuestions

    def chooseNQuestions(self):
        """Method that handles last part in the pipeline. Randomly picks question type
        and randomly picks a question from the generated questions to print out.

        Parameters
        ----------
        Returns
        -------
        """
        generatedQuestions = self.score_questions()
        # score questions and sort them in order

        question_types = list(self.questionsGenerated.keys())
        printed_questions = set()
        while self.nquestions > 0:

            if len(question_types) == 0:
                print("Unable to generate more questions")
                break

            copy_question_types = copy.deepcopy(question_types)
            for current_question_type in copy_question_types:
                try:
                    if self.nquestions <= 0:
                        break

                    current_questions_set = generatedQuestions.get(
                        current_question_type, None)

                    if current_questions_set is None:
                        continue

                    if len(current_questions_set) <= 0:
                        generatedQuestions.pop(
                            current_question_type, None)
                        question_types.remove(current_question_type)
                    else:
                        pick_question = current_questions_set.pop()

                        find_similar_questions = process.extract(
                            pick_question, printed_questions, limit=2)

                        if len(find_similar_questions) > 0 and find_similar_questions[0][1] >= 90:
                            continue

                        else:
                            print(pick_question)

                            printed_questions.add(pick_question)

                            self.nquestions -= 1
                except:
                    pass

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
        for q_type, questions in self.questionsGenerated.items():
            if TYPE is None or q_type == TYPE:
                print(f"---------- PRINTING {q_type} QUESTIONS ----------")
                for q in questions:
                    if q is not None and q != "":
                        print(q)
                print(f"---------- Done with {q_type} QUESTIONS ----------")

    def print_token(self, token):
        """Utility method that prints out information for a spacy token used to debug

        Parameters
        ----------
        token : spacy token
        Returns
        -------
        """
        print(
            f"{token.text:<20}{token.pos_:<20}{token.dep_:<20}{token.ent_type_:<20}{token.head.text:<20}{token.head.dep_:<20}")


if __name__ == "__main__":
    random.seed(11411)
    s = time.time()
    article, nquestions = sys.argv[1], sys.argv[2]

    article = open(article, "r", encoding="UTF-8").read()
    nquestions = int(nquestions)

    ask = Ask(article, nquestions)
    ask.preprocess()

    ask.generateQuestions()
    # ask.printGeneratedQuestions(BINARY)
    # ask.printGeneratedQuestions(WHAT)
    # ask.printGeneratedQuestions(WHO)
    # ask.printGeneratedQuestions(WHERE)
    # ask.printGeneratedQuestions(WHEN)
    # ask.printGeneratedQuestions()  # prints all questions in self.questionsGenerated
    ask.chooseNQuestions()
    e = time.time()
    debugPrint(f"Tried to generate {nquestions}, took {e-s} seconds")
