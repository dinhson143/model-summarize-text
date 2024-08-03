from datetime import datetime

import pandas as pd
import numpy as np
import nltk
import re
import boto3
import pickle

from botocore.exceptions import NoCredentialsError
from gensim.models import KeyedVectors
from tqdm import tqdm
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from rouge_score import rouge_scorer
from joblib import Parallel, delayed

ROOT = "./src/results"
ROOT_DATA = "./src/data"
CURRENT_DATE = datetime.today().date()
BUCKET_NAME = "training-model-summarize-text"
kmeans_file_key = f'kmeans_{CURRENT_DATE}.pkl'
labels_file_key = f'label_{CURRENT_DATE}.pkl'
results_file_key = f'result_train_test_{CURRENT_DATE}_sentoken.txt'
s3 = boto3.client(
    's3',
    aws_access_key_id='AKIAYMDNZKZVFYRTRERO',
    aws_secret_access_key='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',
    region_name='ap-southeast-1'
)


def reading_data_from_files() -> (pd.DataFrame, pd.DataFrame):
    print('1. Reading data from csv files......')
    try:
        data_train = pd.read_csv(f'{ROOT_DATA}/data_train.csv', encoding="utf-8")
        data_test = pd.read_csv(f'{ROOT_DATA}/test_data.csv', encoding="utf-8")
        print('==> Reading data successfully......')
        return data_train, data_test
    except Exception as e:
        print(f'Reading data failed {e}')
        raise


def check_data(data_train: pd.DataFrame, data_test: pd.DataFrame):
    print('2. Shape train and test data ......')
    print("The number of row and column of data_train:", data_train.shape)
    print("The number of row and column of data_test:", data_test.shape)
    print('==> Shape train and test data successfully......')


def create_tokens_from_Sentences(data: pd.DataFrame) -> list:
    nltk.download('punkt')
    print('4. Sentences tokenization.....Create token')

    paras = []
    reg = "[^\\w\\s]"

    print('==> Sentences tokenization => data_train, test.....')
    for i in data.index:
        normalized_text = data.loc[i, 'original'].lower().replace('\n', ' ').strip()
        data.loc[i, 'original'] = normalized_text

        paras_ = nltk.sent_tokenize(data.original[i])
        # Tokenize token
        for j in range(len(paras_)):
            paras_[j] = re.sub(reg, ' ', paras_[j])
            paras_[j] = paras_[j].replace('   ', ' ').replace('  ', ' ')
            paras_[j] = paras_[j].strip()
        paras.append(paras_)
    return paras


def tokenize_text(text):
    return ViTokenizer.tokenize(text).split(" ")


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


def convert_sentence_to_vector(paras: list) -> list:
    print('5.Sentences => Embedding.....Convert sentences to vector')

    fasttext = KeyedVectors.load_word2vec_format(f'{ROOT_DATA}/cc.vi.300.vec')
    vocab = fasttext.key_to_index  # Danh sách các từ trong từ điển

    # Tokenize and vectorize sentences with TF-IDF weights
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_text, vocabulary=vocab)
    # Fit the TF-IDF vectorizer
    tfidf_vectorizer.fit([" ".join(tokenize_text(sentence)) for para in paras for sentence in para])

    # paras_encode = []
    # for para in tqdm(paras, desc="Processing sentences", leave=False):
    #     sentence_vectors = process_paragraph(para, tfidf_vectorizer, fasttext)
    #     paras_encode.append(sentence_vectors)
    #     break
    paras_encode = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_paragraph)(para, tfidf_vectorizer, fasttext) for para in tqdm(paras, desc="Processing sentences", leave=False)
    )

    print('==> Sentences => Embedding.....Convert sentences to vector successfully\n')
    return paras_encode


def train_model(paras_encode: list, paras: list) -> (list, KMeans):
    result = []

    for i in tqdm(range(len(paras_encode)), desc="Processing sentences", leave=False):
        X = paras_encode[i]
        kmeans = None

        # Thử các số cụm khác nhau
        for n_clusters in [4, 3, 2]:
            try:
                kmeans = KMeans(n_clusters=n_clusters).fit(X)
                break
            except Exception as ex:
                raise ex

        if kmeans is None:
            result.append(paras[i])
            continue

        # Tính trung bình các chỉ số của cụm
        avg = [np.mean(np.where(kmeans.labels_ == j)[0]) for j in range(kmeans.n_clusters)]

        # Tìm câu gần nhất với tâm cụm
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        ordering = sorted(range(kmeans.n_clusters), key=lambda k: avg[k])

        # Tạo tóm tắt
        summary = ' '.join([paras[i][closest[idx]] for idx in ordering])
        result.append(summary)
    print(len(result))
    return result, kmeans


def save_model(kmeans: KMeans):
    print('7. Saving model.....')

    kmeans_local_path = f'{ROOT}/kmeans_{CURRENT_DATE}.pkl'
    labels_local_path = f'{ROOT}/label_{CURRENT_DATE}.pkl'

    with open(kmeans_local_path, "wb") as f:
        pickle.dump(kmeans, f)
    with open(labels_local_path, "wb") as f:
        pickle.dump(kmeans.labels_, f)
    print('==> Saving model successfully')

    try:
        s3.upload_file(kmeans_local_path, BUCKET_NAME, kmeans_file_key)
        s3.upload_file(labels_local_path, BUCKET_NAME, labels_file_key)
        print('==> Uploaded model to S3 successfully')
    except FileNotFoundError as e:
        print(f'The file was not found: {e}')
    except NoCredentialsError as e:
        print(f'Credentials not available: {e}')
    except Exception as e:
        print(f'Failed push files to s3: {e}')


def result_train_model(result: list, data_train: pd.DataFrame, data_test: pd.DataFrame):
    print('8. Result training model.....')

    # 8.1 Writing result to file
    print('8.1 Writing result to file========')
    output_path = f"{ROOT}/result_train_test_{CURRENT_DATE}_sentoken.txt"
    with open(output_path, "w", encoding="utf-8") as output:
        for item in result:
            output.write(f"{item}\n")
    print('==> Writing result to file successfully========')
    try:
        s3.upload_file(output_path, BUCKET_NAME, results_file_key)
        print('==> Uploaded model to S3 successfully')
    except FileNotFoundError as e:
        print(f'The file was not found: {e}')
    except NoCredentialsError as e:
        print(f'Credentials not available: {e}')
    except Exception as e:
        print(f'Failed push files to s3: {e}')

    # 8.2 Reading result from file
    print('8.2 Reading result from file ========')
    lines_train_test = []
    with open(output_path, encoding="utf-8") as file:
        lines_train_test = [line.strip() for line in file]
    print('==> Result training model is done')
    print(f'lines_train_test: {lines_train_test}')
    print(f'len(data_train): {len(data_train)}')
    print(f'len(data_test): {len(data_test)}')

    # 8.3 Calculate ROUGE score for train data
    print('8.3 Calculate ROUGE score for train data ========')
    calculate_rouge(data_train, lines_train_test=lines_train_test)

    # 8.4 Calculate ROUGE score for test data
    print('8.4 Calculate ROUGE score for test data ========')
    calculate_rouge(data_test, start_idx=len(data_train), lines_train_test=lines_train_test)

    print('==> All processes completed successfully')


def calculate_rouge(data: pd.DataFrame = None, start_idx: int = 0, lines_train_test: list = None):
    rouge_1, rouge_2, rouge_L = [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i in data.index:
        if i + start_idx < len(lines_train_test):
            scores = scorer.score(data.summary[i], lines_train_test[i + start_idx])
            rouge_1.append(list(scores['rouge1'][0:3]))
            rouge_2.append(list(scores['rouge2'][0:3]))
            rouge_L.append(list(scores['rougeL'][0:3]))

    rouge_1 = pd.DataFrame(rouge_1, columns=['precision', 'recall', 'fmeasure'])
    rouge_2 = pd.DataFrame(rouge_2, columns=['precision', 'recall', 'fmeasure'])
    rouge_L = pd.DataFrame(rouge_L, columns=['precision', 'recall', 'fmeasure'])

    for metric in ['precision', 'recall', 'fmeasure']:
        print(f'File {metric} score')
        print(f'Rouge_1: {rouge_1[metric].mean() * 100}')
        print(f'Rouge_2: {rouge_2[metric].mean() * 100}')
        print(f'Rouge_L: {rouge_L[metric].mean() * 100} \n')

