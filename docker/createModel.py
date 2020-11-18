from sentence_transformers import SentenceTransformer  # Pip installed
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "deepset/bert-base-cased-squad2")
model = AutoModelForQuestionAnswering.from_pretrained(
    "deepset/bert-base-cased-squad2")
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
print("model downloaded")