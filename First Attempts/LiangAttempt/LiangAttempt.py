import sys
from gensim.models import word2vec
import re

if __name__ == '__main__':

    trainIn = sys.argv[1]
    testIn = sys.argv[2]
    train = open(trainIn,"r").read()
    test = open(testIn,"r").read()

    train = re.sub(r"[\s\n\t]+"," ",train)
    test = re.sub(r"[\s\n\t]+"," ",test)
    train = re.sub(r"\.([\s\n\t]+)",".",train)
    test = re.sub(r"\.([\s\n\t]+)",".",train)

    # convert training data into proper input
    train = train.split(".")
    for i in range(len(train)):
    	train[i] = train[i].split(" ")

    question = input("Please enter your question: ")

    # strip question and then remove question mark if its there
    question = question.strip()
    if len(question) != 0 and question[len(question)-1] == "?":
    	question = question[:len(question)-1]

    model = word2vec.Word2Vec(train, min_count = 1)

    print(model.wv.most_similar(['Indus',"Valley"]))


