'''
Class that will return a set of questions based on given text
Command: ./ask article.txt nquestions
Python command: python answer.py article.txt 21
'''

import sys

import numpy as np
import spacy

from parse import Parse

class Ask:
    def __init__(self, article, nquestions):
        self.article = article
        self.nquestions = nquestions
        self.nlp = spacy.load("en_core_web_md")  # spacy model
        self.preprocess()

    def preprocess(self):
        p = Parse()

        self.corpus = p.parseCorpus(self.article)
        self.spacyCorpus = self.nlp(self.corpus)

    def generateWhenQuestions(self):
        """
        create when questions
        
        Returns: list of when questions
        """

        for sent in self.spacyCorpus.sents:
            # print(sent)
            # print([(ent, ent.label_) for ent  in sent.ents])
            foundDate, foundEvent = False, False
            whenQuestions = []
            for ent in sent.ents:
                currQuestion, obj = ["When", "[auxVerb]"], None
                if ent.label_ in set(["DATE"]):
                    print("Sentence: {}".format(sent))
                    print("Entity:", ent, ent.label_)
                    foundDate = True
            if foundDate:
                # print([(tok, tok.pos, tok.dep_) for tok in sent])
                print("Chunks", [c for c in sent.noun_chunks])
                plural = False # default to singular
                for chunk in sent.noun_chunks:
                    
                    if chunk.root.dep_  == "nsubj":
                        currQuestion.append(chunk.text)
                        currQuestion.append(chunk.root.head.lemma_)
                        rootVerb = chunk.root.head
                        #plural vs singular
                        if chunk.root.tag_ == "NNPS" or chunk.root.tag_ == "NNS":
                            plural = True
                        tagMap = self.nlp.vocab.morphology.tag_map[chunk.root.head.tag_]
                        pastTense = False #default to present tense
                        print(chunk.root.head.text, tagMap)
                        if "Tense_past" in tagMap and tagMap["Tense_past"] == True:
                            pastTense = True
                        
                        

                    if chunk.root.dep_ == "dobj" and chunk.root.head.text == rootVerb.text:
                        obj = chunk.text
                        
                if obj is not None:
                    currQuestion.append(obj)
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
                    currQuestion.insert(1, rootVerb)
                else:
                    if pastTense:
                        conjugatedVerb = "did"
                    else: #presentTense
                        if plural:
                            conjugatedVerb = "do"
                        else: #singular
                            conjugatedVerb = "did"
                        
                    if "[auxVerb]" in currQuestion:
                        currQuestion.remove("[auxVerb]")
                    currQuestion.insert(1, conjugatedVerb)
                if len(currQuestion) > 2:
                    print(currQuestion)
            print("\n")

    # just exploring trying to parse  dep tree
    def generateWhenQuestionDepTree(self):

        for sent in self.spacyCorpus.sents:
            # print(sent)
            # print([(ent, ent.label_) for ent  in sent.ents])
            foundDate, foundEvent = False, False
            whenQuestions = []
            for ent in sent.ents:
                currQuestion, obj = ["When", "[auxVerb]"], None
                if ent.label_ in set(["DATE"]):
                    print("Sentence: {}".format(sent))
                    # print("Entity:", ent, ent.label_)
                    foundDate = True

            if foundDate:
                # print([psubj for psubj in sent])
                for possible_subject in sent:
                    
                    if possible_subject.dep_ == "nsubj" and possible_subject.head.pos_ == "VERB":
                        currQuestion.append(([l for l in possible_subject.lefts], "left"))
                        currQuestion.append((possible_subject, "nsubj"))
                        currQuestion.append((possible_subject.head, "VERB"))

                        #if possible_subject.head.lemma_ == "be":
                            #is - singular, present tense
                            
                            #am - subject == I, singular, present tense
                            #are - subject == you OR plural, present tense
                            #was - subject = singular, past tense
                            #were - subject = plural, past tense
                        #else:
                            
                            #past: did
                            #singular: does
                            #plural: do
                            #currQuestion[1] = "do"
                        break
                        
                if len(currQuestion) < 4:
                    break 
                print(currQuestion)
            print("\n")
    def generateQuestions(self):
        pass

    def chooseNQuestions(self):
        pass


if __name__ == "__main__":
    article, nquestions = "a1.txt", 1#sys.argv[1], sys.argv[2]

    article = open(article, "r", encoding="UTF-8").read()
    nquestions = int(nquestions)

    ask = Ask(article, nquestions)
    ask.preprocess()
    ask.generateWhenQuestions()
    # ask.generateWhenQuestionDepTree()
