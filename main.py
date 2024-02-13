import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from check_submission import compare_files
import time
import pandas as pd


def train_model(feature_classes, train_path, threshold, lam, weights_path, retrain=True):
    """
    Train a model and save it in a pickle.
    """
    start_time = time.time()
    if retrain:
        statistics, feature2id = preprocess_train(train_path, threshold, feature_classes)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    train_time = time.time() - start_time
    return feature2id, pre_trained_weights, train_time


def test_model(feature2id, pre_trained_weights, test_path, predictions_path, return_confusion_matrix=False):
    """
    Test a model on a test file
    """
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    return compare_files(test_path, predictions_path, return_confusion_matrix)


def cross_validation(feature_classes, train_path, threshold, lam, weights_path, predictions_path, folds_num):
    """
    Cross validate over train text file
    """
    sum_accuracies = 0
    cross_train_path = "data/train3.wtag"
    val_path = "data/test2.wtag"
    for i in range(folds_num):
        # divide to train and validate files
        with open(train_path, 'r') as f:
            lines = f.readlines()
            train_lines = lines[:int(i*len(lines)/folds_num)] + lines[int((i+1)*len(lines)/folds_num):]
            val_lines = lines[int(i*len(lines)/folds_num): int((i+1)*len(lines)/folds_num)]
        with open(cross_train_path, 'w') as f:
            f.writelines(train_lines)
        with open(val_path, 'w') as f:
            f.writelines(val_lines)
        # validate configuration
        feature2id, pre_trained_weights, _ = train_model(feature_classes, cross_train_path,
                                                         threshold, lam, weights_path)
        accuracy = test_model(feature2id, pre_trained_weights, val_path, predictions_path)[0]
        sum_accuracies += accuracy
    return sum_accuracies / folds_num


def main():
    # constants
    feature_classes1 = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "capital", "number",
                       "contains hyphen", "pp_word", "f100_lower", "f101_lower", "f102_lower", "f106_lower",
                       "f107_lower", "nn_word", "c_word_n_word", "length", "2words", "3words"]

    feature_classes2 = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "capital", "number",
                       "contains hyphen", "pp_word", "length"]
    threshold = 1
    lam = 1

    train1_path = "data/train1.wtag"
    train2_path = "data/train2.wtag"
    test_path = "data/test1.wtag"

    weights1_path = 'weights1.pkl'
    weights2_path = 'weights2.pkl'
    predictions_path = 'predictions.wtag'

    # Training
    print("Train model 1")
    feature2id1, pre_trained_weights1, train_time1 = train_model(feature_classes1, train1_path,
                                                                threshold, lam, weights1_path, retrain=True)
    print("Calc model 1 train accuracy")
    train_accuracy1 = test_model(feature2id1, pre_trained_weights1, train1_path, predictions_path)[0]

    print(f"Train 1 accuracy: {train_accuracy1}")
    print(f"Train 1 time: {train_time1}")

    print("Train model 2")
    feature2id2, pre_trained_weights2, train_time2 = train_model(feature_classes2, train2_path,
                                                                threshold, lam, weights2_path, retrain=True)
    print("Calc model 2 train accuracy")
    train_accuracy2 = test_model(feature2id2, pre_trained_weights2, train2_path, predictions_path)[0]

    print(f"Train 2 accuracy: {train_accuracy2}")
    print(f"Train 2 time: {train_time2}")

    # Test model 1
    test_results = test_model(feature2id1, pre_trained_weights1, test_path, predictions_path,
                              return_confusion_matrix=True)
    test_accuracy, _, conf_mat = test_results
    print(f"Test 1 accuracy: {test_accuracy}")
    # find worst tags
    row_sum = conf_mat.sum(axis=1)
    diag = conf_mat.apply(lambda row: row[row.name])
    mistakes = row_sum.sub(diag)
    top10mistakes_tags = list(mistakes.nlargest(10).index)
    worst_tags_conf_mat = conf_mat.loc[top10mistakes_tags][top10mistakes_tags]
    print(worst_tags_conf_mat)

    # Test model 2
    print("Cross validate model 2")
    cross_val_accuracy = cross_validation(feature_classes2, train2_path, threshold, lam,
                                          weights2_path, predictions_path, 5)
    print(f"Cross validation accuracy: {cross_val_accuracy}")


if __name__ == '__main__':
    main()
