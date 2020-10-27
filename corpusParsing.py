import re
import sys

# Corpus Parsing to run this follow the command "python corpusParsing.py corpusPath.txt"
# where corpusPath.txt is the path to the corpus text you are reading in
# an example text to test is Development_data/set1/a1.txt

if __name__ == '__main__':

	# takes in corpus text and reads it in as a string
    corpusIn = sys.argv[1]
    corpus = open(corpusIn,"r").read()
    corpus = corpus.strip()

    # convert to lower case, might not want to do this for parsing later
    # corpus = corpus.lower()

    # convert multiple new lines into a single new line
    corpus = re.sub(r"[\n]+","\n",corpus)

    # convert periods with a bunch of spaces/newlines/tabs after to just a period
    # however does not convert if the period and spaces/newlines/tabs is followed by the ) character
    # this helps not filter out cases where the sentence is like (c. 2686-2181 BC)
    corpus = re.sub(r"\.([\s\n\t]+)(?![^\.\(]*\))",".",corpus)

    # convert the new lines into periods
    corpus = re.sub(r"[\n]+",".",corpus)

    # convert the spaces and tabs into single space
    corpus = re.sub(r"[\s\t]+"," ",corpus)

    # convert corpus into a vector of sentences and create another copy for word vector later
    # however does not split if the period is followed by a space and then the ) character
    # this helps not split on cases where the sentence is like (c. 2686-2181 BC)
    corpSen = re.split(r"\.(?! [^\.\(]*\))",corpus)
    #corpWords = re.split(r"\.(?! [^\.\(]*\))",corpus)

    # convert each vector of sentences into vector of words
    # result is a vector of vectors
    #for i in range(len(corpWords)):
    #	corpWords[i] = corpWords[i].split(" ")

    print(corpSen[1])

    # We are including references at bottom of each wiki corpus, this is something we may want to remove in the future
    # if it affects our algorithm later

