import re
import sys

# Corpus Parsing to run this follow the command "python corpusParsing.py corpusPath.txt"
# where corpusPath.txt is the path to the corpus text you are reading in
# an example text to test is Development_data/set1/a1.txt

if __name__ == '__main__':

	# takes in corpus text and reads it in as a string
    corpusIn = sys.argv[1]
    corpus = open(corpusIn,"r").read()

    # convert to lower case, might not want to do this for parsing later
    # corpus = corpus.lower()

    # convert multiple new lines into a single new line
    corpus = re.sub(r"[\n]+","\n",corpus)

    # convert periods with a bunch of spaces/newlines/tabs after to just a period
    corpus = re.sub(r"\.([\s\n\t]+)",".",corpus)

    # convert the new lines into periods
    corpus = re.sub(r"[\n]+",".",corpus)

    # convert the spaces and tabs into single space
    corpus = re.sub(r"[\s\t]+"," ",corpus)

    # convert corpus into a vector of sentences and create another copy for word vector later
    corpSen = corpus.split(".")
    corpWords = corpus.split(".")

    # convert each vector of sentences into vector of words
    for i in range(len(corpWords)):
    	corpWords[i] = corpWords[i].split(" ")

    print(corpWords[0])

