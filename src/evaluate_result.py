from datetime import datetime

import boto3
import pandas as pd
from rouge_score import rouge_scorer

ROOT = "./results"
ROOT_DATA = "./data"
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


def reading_data_from_files_ev() -> (pd.DataFrame, pd.DataFrame):
    print('1. Reading data from csv files......')
    try:
        data_train_or = pd.read_csv(f'{ROOT_DATA}/data_train.csv', encoding="utf-8")
        data_test_or = pd.read_csv(f'{ROOT_DATA}/test_data.csv', encoding="utf-8")
        data_train = data_train_or.sample(n=2400, random_state=42)
        data_test = data_test_or.sample(n=600, random_state=42)
        print('==> Reading data successfully......')
        return data_train, data_test
    except Exception as e:
        print(f'Reading data failed {e}')
        raise


def result_train_model_ev(data_train: pd.DataFrame, data_test: pd.DataFrame):
    print('8. Result training model.....')

    # 8.1 Writing result to file
    print('8.1 Writing result to file========')
    output_path = "./results/result_train_test_2024-07-29_sentoken.txt"

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
    calculate_rouge_ev(data_train, lines_train_test=lines_train_test)

    # 8.4 Calculate ROUGE score for test data
    print('8.4 Calculate ROUGE score for test data ========')
    calculate_rouge_ev(data_test, start_idx=len(data_train), lines_train_test=lines_train_test)

    print('==> All processes completed successfully')


def calculate_rouge_ev(data: pd.DataFrame = None, start_idx: int = 0, lines_train_test: list = None):
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

if __name__ == '__main__':
    print("Reading data from files...")
    data_train, data_test = reading_data_from_files_ev()

    # Đánh giá kết quả huấn luyện mô hình
    print("Evaluating training results...")
    result_train_model_ev(data_train, data_test)

