
import nltk
import numpy as np
import torch
from InferSent.models import InferSent

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def preprocessQs(s):
    # RN just drop the ?, assuming its the last char
    s = s[0:-1]
    return s

if __name__ == "__main__":
    text = "Development_data/set1/a1.txt"
    questions = "q.txt"
    sentTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(text, "r+") as f:
        # print(sentTokenizer.tokenize(f.read().strip()))
        sentences = sentTokenizer.tokenize(f.read().strip())
    ### pasted from https://github.com/facebookresearch/InferSent

    
    V = 2
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = 'fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)

    # infersent.build_vocab(sentences, tokenize=True)
    infersent.build_vocab_k_words(K=100000)


    embeddings = infersent.encode(sentences, bsize=128, tokenize=False, verbose=True)
    print('nb sentences encoded : {0}'.format(len(embeddings)))
    #### End Paste

    parsedQs = []
    with open(questions, "r+") as f:
        for q in f.readlines():
            parsedQs.append(preprocessQs(q))
    # print(parsedQs)s

    qEmbeddings = infersent.encode(parsedQs, bsize=128, tokenize=False, verbose=True)
    print('nb sentences encoded : {0}'.format(len(embeddings)))
    print(qEmbeddings.shape) # Should be (# of questions, 4096)

    # np.apply_along_axis(function, 1, array)
    # print(embeddings.shape)
    # lets find the sentence that is msot close to the first question
    # print(cosine(qEmbeddings[0], embeddings[0]))
    q1Cos = np.apply_along_axis(cosine, 1, embeddings, qEmbeddings[0])
    print(q1Cos.shape) # should be (number of sentences in our text, )

    print("Question:", parsedQs[0])
    print("Answer:",sentences[np.argmax(q1Cos)], sep = "\n")

    # Same procedure for Q2, can be a helper
    q2Cos = np.apply_along_axis(cosine, 1, embeddings, qEmbeddings[2])

    print("Question:", parsedQs[2])
    print("Answer:",sentences[np.argmax(q2Cos)], sep = "\n")