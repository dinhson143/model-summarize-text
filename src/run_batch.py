from src.main import reading_data_from_files, check_data, create_tokens_from_Sentences, convert_sentence_to_vector, \
    save_model, result_train_model, train_model


def run_batch():
    print("Reading data from files...")
    data_train, data_test = reading_data_from_files()

    # Kiểm tra dữ liệu
    print("Checking data...")
    check_data(data_train, data_test)

    import random
    # Tạo tokens từ sentences
    print("Creating tokens from sentences...")
    paras_train = create_tokens_from_Sentences(data_train)
    paras_test = create_tokens_from_Sentences(data_test)
    paras = []
    # Combine the samples
    paras.extend(paras_train)
    paras.extend(paras_test)  # Kết hợp các tokens

    # Chuyển đổi câu thành vector
    print("Converting sentences to vectors...")
    paras_encode = convert_sentence_to_vector(paras)

    # Huấn luyện mô hình
    print("Training model...")
    result, kmeans = train_model(paras_encode, paras)

    # Lưu mô hình
    print("Saving model...")
    save_model(kmeans)

    # Đánh giá kết quả huấn luyện mô hình
    print("Evaluating training results...")
    result_train_model(result, data_train, data_test)


if __name__ == '__main__':
    run_batch()