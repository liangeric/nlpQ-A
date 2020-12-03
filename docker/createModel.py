from sentence_transformers import SentenceTransformer  # Pip installed
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
model1 = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
model2 = SentenceTransformer("distilroberta-base-msmarco-v2")
# print("model downloaded")