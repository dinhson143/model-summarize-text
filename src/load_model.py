import numpy as np
import pandas as pd
import nltk
import re
import pickle

from gensim.models import KeyedVectors
from tqdm import tqdm
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min

ROOT = "./results"
ROOT_DATA = "./data"


def tfidf_weighted_vector(sentence, tfidf_vectorizer, fasttext):
    words = tokenize_text(sentence)
    tfidf_vector = tfidf_vectorizer.transform([" ".join(words)])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_vector.toarray().flatten()

    vector = np.zeros(fasttext.vector_size)
    total_weight = 0.0

    for word, score in zip(feature_names, tfidf_scores):
        if word in fasttext.key_to_index:
            vector += fasttext[word] * score
            total_weight += score

    if total_weight > 0:
        vector /= total_weight

    return vector


def process_paragraph(para, tfidf_vectorizer, fasttext):
    sentence_vectors = []
    for sentence in para:
        words = sentence.split(" ")
        sentence_vector = tfidf_weighted_vector(sentence, tfidf_vectorizer, fasttext)
        sentence_vector = sentence_vector / len(words)
        sentence_vectors.append(sentence_vector)
    return sentence_vectors


def tokenize_text(text):
    return ViTokenizer.tokenize(text).split(" ")


def convert_sentence_to_vector(paras: list) -> list:
    fasttext = KeyedVectors.load_word2vec_format(f'{ROOT_DATA}/cc.vi.300.vec')
    vocab = fasttext.key_to_index  # Danh sách các từ trong từ điển

    # Tokenize and vectorize sentences with TF-IDF weights
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_text, vocabulary=vocab)
    # Fit the TF-IDF vectorizer
    tfidf_vectorizer.fit([" ".join(tokenize_text(sentence)) for para in paras for sentence in para])

    paras_encode = []
    for para in tqdm(paras, desc="Processing sentences", leave=False):
        sentence_vectors = process_paragraph(para, tfidf_vectorizer, fasttext)
        paras_encode.append(sentence_vectors)

    # index = 0
    # paras_encode = Parallel(n_jobs=-1, backend="multiprocessing")(
    #     delayed(process_paragraph)(index, para, tfidf_vectorizer, fasttext) for para in tqdm(paras, desc="Processing sentences", leave=False)
    # )

    print('==> Sentences => Embedding.....Convert sentences to vector successfully\n')
    return paras_encode


if __name__ == '__main__':
    nltk.download('punkt')

    # load data
    print('1. Reading data need to be predicted......')
    data_du_bao = pd.read_csv(f'{ROOT_DATA}/datamau2.csv')
    print('==> Reading data need to be predicted successfully......')

    print('2. Shape train and test data ......')
    print("==> The number of row and column of data_dubao:", data_du_bao.shape)

    print('3. Sentences tokenization.....Create token')
    paras_du_bao = []
    reg = "[^\\w\\s]"
    print('==> Sentences tokenization => data_train, test.....')
    for i in data_du_bao.index:
        normalized_text = data_du_bao.loc[i, 'original'].lower().replace('\n', ' ').strip()
        print(normalized_text)
        data_du_bao.loc[i, 'original'] = normalized_text
        paras_ = nltk.sent_tokenize(data_du_bao.original[i])
        # Tokenize token
        for j in range(len(paras_)):
            paras_[j] = re.sub(reg, ' ', paras_[j])
            paras_[j] = paras_[j].replace('   ', ' ').replace('  ', ' ')
            paras_[j] = paras_[j].strip()
        paras_du_bao.append(paras_)
    print('==> Sentences tokenization successfully.\n')

    print("4. Converting sentences to vectors...")
    paras_encode = convert_sentence_to_vector(paras_du_bao)

    print('5. Loading model Kmeans')
    with open(f'{ROOT}/kmeans_2024-07-29.pkl', "rb") as f:
        kmeans = pickle.load(f)
    with open(f'{ROOT}/label_2024-07-29.pkl', "rb") as f:
        kmeans.labels_ = pickle.load(f)
    print('==> Load model Kmeans successfully')

    print('6. Summarize data du bao:')
    result_du_bao = []
    for i in range(len(paras_encode)):
        X = paras_encode[i]

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        summary = ' '.join([paras_du_bao[i][idx] for idx in closest])
        result_du_bao.append(summary)
    print(result_du_bao)
    print('==> Summarize data dubao successfully')