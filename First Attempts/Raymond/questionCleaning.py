import spacy 

'''
Reads questions file, and return list of tokens, drop WH word and "?"
fp str of filepath to the questions.txt
model spacy object, for the model
excluded_tokens a python set for words to remove

Current implmentation 
'''
def parseQuestions(fp, model, excluded_tokens):
    # cleanQs = []
    # with open(fp, "r+") as f:
    #     for sent in f.read().split("\n"):
    #         cleanSents_string = []
    #         for word in sent.split(" "):
    #         # print(word.upper())
    #             if "?" in word:
    #                 word = word.replace("?", "")
    #             elif word.upper() not in excluded_tokens:
    #                 cleanSents_string.append(word)
    #         cleanSents_string = " ".join(cleanSents_string)
    #         cleanQs.append(cleanSents_string)
    #     cleanQs = "\n".join(cleanQs)
    with open(fp, "r+") as f:
        rawQs = f.read()  # read in the text as whole string
    spacyQs = spacyModel(rawQs)  # call the model on the raw strings 
    for sent in spacyQs.sents:
        print(sent)
        

if __name__ == "__main__":
    spacyModel = spacy.load("en_core_web_md")
    # exclude = set(["Who", "What", "When", "Where", "How", "?"])
    exclude = set(["WHO", "WHAT", "WHEN", "WHERE", "HOW", "?"])

    parseQuestions("q.txt", spacyModel, exclude)







