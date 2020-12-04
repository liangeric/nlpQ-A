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


from fuzzywuzzy import process

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
        self.nlp = spacy.load("en_core_web_lg")  # spacy model

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
        # given sentence is not grammatically correct and has date/time
        rootVerb = None
        whenFound = False
        whenClause = set()
        for tok in sent:
            if tok.dep_ is not None and tok.dep_ == "ROOT":
                rootVerb = tok
            if whenFound and (tok.ent_type is None or tok.ent_type not in set(["DATE", "TIME"])):
                whenFound = False
            if tok.ent_type is not None and tok.ent_type_ in set(["DATE", "TIME"]):
                if whenFound == False:
                    whenVerb = tok  # initialize verb to be date/time token
                    whenFound = True
                whenClause.add(tok)

        if rootVerb is None or len(whenClause) == 0:
            return

        # else: make a question
        whenSubj = None
        stopSent_i = 0
        iteration = 0
        maxIter = len(sent)
        while whenVerb.pos_ != "VERB":
            if iteration == maxIter:
                return
            if whenVerb.head.pos_ == "VERB":
                whenVerbChild = whenVerb
            whenVerb = whenVerb.head
            iteration += 1

        plural = False  # default to singular
        pastTense = False # default to present
        firstChunk = True
        whenObj = []
        rootVerbChildren = []
        for child in whenVerb.children:

            if child.dep_ == "nsubj":
                # plural vs singular
                if firstChunk and child.tag_ == "NNPS" or child.tag_ == "NNS":
                    plural = True
                tagMap = self.nlp.vocab.morphology.tag_map[whenVerb.tag_]
                if firstChunk and "Tense_past" in tagMap and tagMap["Tense_past"] == True:
                    pastTense = True
                firstChunk = False

                whenSubj = child
            rootVerbChildren.append(child)

        for chunk in sent.noun_chunks:
            # don't want to add the when chunk
            whenTermFound = False
            for whenTerm in whenClause:
                if whenTerm.text in chunk.text:
                    whenTermFound = True
            if whenTermFound or whenVerbChild.text in chunk.root.head.text:
                continue
            if chunk.root.dep_ == "nsubj" and chunk.root.head == whenVerb:
                whenSubj = chunk
            elif chunk.root.dep_ == "dobj" and chunk.root.head == whenVerb:
                whenObj.append(chunk)
            elif chunk.root.dep_ == "pobj":
                if chunk.root.head.head is not None and len(whenObj) > 0 and chunk.root.head.head == whenObj[-1].root:
                    whenObj.append(chunk.root.head)
                    whenObj.append(chunk)

        if whenSubj is None:
            return

        currQuestion = ["When", "[auxVerb]"]
        if whenVerb.lemma_ == "be":
            currQuestion = currQuestion[:-1]
            currQuestion.append(whenVerb.text)
            currQuestion.append(whenSubj.text)

            if "[auxVerb]" in currQuestion:
                currQuestion.remove("[auxVerb]")

        else:
            if pastTense:
                conjugatedVerb = "did"
            else:  # presentTense
                if plural:
                    conjugatedVerb = "do"
                else:  # singular
                    conjugatedVerb = "does"
            if "[auxVerb]" in currQuestion:
                currQuestion.remove("[auxVerb]")
            currQuestion.insert(1, conjugatedVerb)
            currQuestion.append(whenSubj.text)
            currQuestion.append(whenVerb.lemma_)
            for chunk in whenObj:
                currQuestion.append(chunk.text)
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
            if token.pos_ == "AUX" and token.dep_ in ["ROOT", "ccomp"]:
                for child in token.children:
                    if child.dep_ == "nsubj":
                        child_subtree = list(child.subtree)
                        if len(child_subtree) > 0:
                            first, last = child_subtree[0], child_subtree[-1]
                            subj = ''.join(
                                t.text_with_ws for t in self.spacyCorpus[first.i: last.i + 1])

                            question_word = token.text.capitalize()
                            if question_word.strip().lower() not in ['is', 'was', 'are', 'were']:
                                continue
                            question_body = subj + " " + \
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

        whereChunk = []
        whereVerbList = []
        for chunk in sent.noun_chunks:
            if chunk.root.ent_type_ in set(["GPE", "LOC"]) and chunk.root.dep_ == "pobj":
                whereChunk.append(chunk)
                whereVerbList.append(chunk.root) #initialize whereVerb to be where root token


        if len(whereChunk) == 0:
            return
        for chunk_i in range(len(whereChunk)): # possibly make a where question for each where chunk found
            qChunk = whereChunk[chunk_i]
            whereVerb = whereVerbList[chunk_i]
            #make question
            whereSubj = None
            iteration = 0
            maxIter = len(sent)
            while whereVerb.pos_ != "VERB":
                if iteration == maxIter:
                    return
                if whereVerb.head.pos_ == "VERB":
                    whereVerbChild = whereVerb
                whereVerb = whereVerb.head
                iteration += 1
            plural = False
            pastTense = False
            conjugatedVerb = None
            whereObj = []
            rootVerbChildrenExceptWhere = []
            for child in whereVerb.children:
                if child.dep_ == "nsubj":
                    if child.tag_ == "NNPS" or child.tag_ == "NNS":
                        plural = True
                    tagMap = self.nlp.vocab.morphology.tag_map[whereVerb.tag_]
                    if "Tense_past" in tagMap and tagMap["Tense_past"] == True:
                        pastTense = True
                    whereSubj = child
                elif chunk.root.dep == "aux":
                    conjugatedVerb = chunk
                if child != whereVerbChild: #don't want the where prep phrase
                    rootVerbChildrenExceptWhere.append(child)

            for chunk in sent.noun_chunks:
                whereTermFound = False
                if qChunk.text == chunk.text:
                    whereTermFound = True
                if whereTermFound or whereVerbChild.text in chunk.root.head.text:
                    continue
                
                if chunk.root.dep_ == "nsubj" and chunk.root.head == whereVerb:
                    whereSubj = chunk
                elif chunk.root.dep_ == "dobj" and chunk.root.head == whereVerb:
                    whereObj.append(chunk)
                elif chunk.root.dep_ == "pobj":
                    if chunk.root.head.head is not None and len(whereObj) > 0 and chunk.root.head.head == whereObj[-1].root:
                        whereObj.append(chunk.root.head)
                        whereObj.append(chunk)
                

            if whereSubj is None:
                return

            currQuestion = ["Where", "[auxVerb]"]
            #this shouldn't happen bc the WHERE is in a prep phrase, but just in case
            if whereVerb.lemma == "be":
                currQuestion = currquestion[:-1]
                currQuestion.append(whereVerb.text)
                currQuestion.append(whereSubj.text)
            elif conjugatedVerb is not None:
                continue
            else:
                if pastTense:
                    conjugatedVerb = "did"
                elif not pastTense: #presentTense
                    if plural:
                        conjugatedVerb = "do"
                    else: #singular
                        conjugatedVerb = "does"
                if "[auxVerb]" in currQuestion:
                    currQuestion.remove("[auxVerb]")
                currQuestion.insert(1, conjugatedVerb)
                currQuestion.append(whereSubj.text)
                currQuestion.append(whereVerb.lemma_)
                for chunk in whereObj:
                    currQuestion.append(chunk.text)
            if len(currQuestion) > 2:
                q = " ".join(currQuestion[1:])
                self.addQuestionToDict(q, WHERE)


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
        good_pos = set(["anybody", "anyone", "everybody", "everyone", "he", "her", "who"
                        "herself", "him", "himself", "I", "me", "no one", "nobody", "she", "she",
                        "somebody", "someone", "they", "them", "us", "thou", "we", "you"])

        for token in sent:
            if token.ent_type_ in ner or token.text.lower() in good_pos:
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
        verbs = set(["AUX", "VERB"])
        ner = set(["ORG", "PRODUCT"])

        for token in sent:
            if token.ent_type_ in ner:
                head = token.head
                if head.pos_ in verbs and head.dep_ == "ROOT":
                    questionType = WHAT

                    q = ''.join(
                        t.text_with_ws for t in self.spacyCorpus[head.i:sent.end-1])

                    qLower = q.lower()
                    tempSplit = qLower.split(" ")

                    if "he" in tempSplit or "she" in tempSplit:
                        questionType = WHO

                    self.addQuestionToDict(q, questionType)

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
            if text.isspace():
                continue
            upper = token.pos_ == "PROPN" or token.ent_type_ in [
                'GPE', 'LOC', 'PERSON', 'DATE', 'ORG', 'PRODUCT']

            if TYPE == BINARY and token.i >= 1 and not upper:
                text = text.lower()

            if TYPE != BINARY and token.i >= 0 and not upper:
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

        for sent in self.spacyCorpus.sents:

            self.generateWhat(sent)
            self.generateWho(sent)
            self.generateWhAux(sent)
            self.generateBinary(sent)
            try:
                self.generateWhen(sent)
            except:
                continue

            self.generateWhere(sent)

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
                question_nlp = self.nlp(q)

                current_score = 0

                for tok in question_nlp:
                    if tok.pos_ == "PRON" and tok.dep_ == "nsubj":
                        current_score -= 5

                question_tokens = q.split(" ")
                ents = question_nlp.ents

                ideal_number_tokens = 12
                diff_ideal_tokens = abs(
                    ideal_number_tokens-len(question_tokens))

                if len(ents) == 0:
                    current_score -= 5

                ideal_number_ents = 1.5
                diff_ideal_ents = abs(ideal_number_ents-len(ents))

                current_score -= diff_ideal_tokens
                current_score -= diff_ideal_ents

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
