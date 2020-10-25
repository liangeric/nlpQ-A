from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  #  Need to separately download this 
from nltk.stem import WordNetLemmatizer # Also need separate download
import spacy

def NLTKWorkFlow(txtFP):
    with open(txtFP, "r+") as f:
        text_tokens = word_tokenize(f.read())
    print("Original Text Tokens")
    print(text_tokens[0:50])
    print("The length of tokens", len(text_tokens))
    
    # Say we want to remove stopping words
    stop_words = set(stopwords.words('english')) 
    removal_criteria = stop_words | set()
    new_tokens = [tok for tok in text_tokens if tok not in removal_criteria]
    print("Length after Stop word Removal", len(new_tokens))
    print(new_tokens[0:50])

    # If we want to do lemmatizing with NLTK we can use WordNet Lemmatizer. 
    # If the word is in WordNet, it will change, otherwise it does nothing. 
    # Lemmatize needs POS ???? bruh 
    wnl = WordNetLemmatizer()
    lemmatized_tokens = []
    for tok in text_tokens:
        new_word = wnl.lemmatize(tok)
        lemmatized_tokens.append(wnl.lemmatize(tok))
    print("Lemmatized tokens")
    print(lemmatized_tokens[0:100])


# Also needs to download and install a specific model i  think
# python -m spacy download en_core_web_sm
def spaCyWorkflow(txtFP, qFP):
    # Loads a model that you should have downloaded and installed 
    # Created a SpaCy doc object if called onto a string
    nlp = spacy.load("en_core_web_sm")

    with open(txtFP, "r+") as f:
        txt_doc = nlp(f.read())
        # This is a very large object since it now holds the whole text
    with open(qFP, "r+") as f:
        question_doc = nlp(f.read())
    # You can loop through the Doc object as below and pull out POS/Dep/ etc. 
    for token in question_doc:
        print(token.text, token.pos_, token.dep_)

if __name__ == "__main__":
    text = "Development_data/set1/a1.txt"
    questions = "q.txt"
    # NLTKWorkFlow(text)
    
    # _______________________________
    # Now SpaCy 
    spaCyWorkflow(text, questions)