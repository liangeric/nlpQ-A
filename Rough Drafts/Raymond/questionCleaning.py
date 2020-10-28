import spacy
import numpy as np


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def getAvgVec(doc, excludedWords):
    avg = []
    for sentence in doc.sents:
        accum = np.zeros((300,))
        for word in sentence:
            if word.text not in excludedWords and not word.is_stop:
                accum += word.vector
        avg.append(accum / len(sentence))
    return avg

# replace w Eric/Samarth's code


def getAvgVecCorpus(fp, spacyModel):
    with open(fp, "r+") as f:
        rawCorpus = f.read()
    corpus = spacyModel(rawCorpus)
    avg = []
    for sentence in corpus.sents:
        accum = np.zeros((300,))
        for word in sentence:
            if not word.is_stop:
                accum += word.vector
        avg.append(accum / len(sentence))
    return avg, corpus


'''
Reads questions file, and return list of tokens, drop WH word and "?"
fp str of filepath to the questions.txt
model spacy object, for the model
excluded_tokens a python set for words to remove

Current implmentation 
'''


def parseQuestions(fp, model, excluded_tokens):
    with open(fp, "r+") as f:
        rawQs = f.read()  # read in the text as whole string
    spacyQs = spacyModel(rawQs)  # call the model on the raw strings
    return getAvgVec(spacyQs, excluded_tokens), spacyQs


if __name__ == "__main__":
    spacyModel = spacy.load("en_core_web_md")
    # exclude = set(["Who", "What", "When", "Where", "How", "?"])
    exclude = set(["WHO", "WHAT", "WHEN", "WHERE", "HOW", "?"])

    qs, questionText = parseQuestions("q.txt", spacyModel, exclude)
    cs, corpus = getAvgVecCorpus("a1.txt", spacyModel)

    for i in range(len(qs)):
        cos = np.apply_along_axis(cosine, 1, cs, qs[i])

        print("Question:", list(questionText.sents)[i])
        print("Answer:", list(corpus.sents)[np.argmax(cos)])
