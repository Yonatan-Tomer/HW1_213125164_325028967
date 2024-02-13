import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from check_submission import compare_files
import time
import pandas as pd


def train_model(feature_classes, train_path, threshold, lam, weights_path, retrain=True):
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
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    return compare_files(test_path, predictions_path, return_confusion_matrix)


def cross_validation(feature_classes, train_path, threshold, lam, weights_path, test_path, predictions_path, folds_num):
    sum_accuracies = 0
    cross_train_path = "data/train3.wtag"
    val_path = "data/test2.wtag"
    for i in range(folds_num):
        with open(train_path, 'r') as f:
            lines = f.readlines()
            train_lines = lines[:int(i*len(lines)/folds_num)] + lines[int((i+1)*len(lines)/folds_num):]
            val_lines = lines[int(i*len(lines)/folds_num): int((i+1)*len(lines)/folds_num)]
        with open(cross_train_path, 'w') as f:
            f.writelines(train_lines)
        with open(val_path, 'w') as f:
            f.writelines(val_lines)

        feature2id, pre_trained_weights, _ = train_model(feature_classes, cross_train_path,
                                                         threshold, lam, weights_path)
        accuracy = test_model(feature2id, pre_trained_weights, val_path, predictions_path)[0]
        sum_accuracies += accuracy
    return sum_accuracies / folds_num



def main():
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
    comp1_path = "data/comp1.words"
    comp2_path ="data/comp2.words"
    comp1_pred_path = "comp1_pred.wtag"
    comp2_pred_path = "comp2_pred.wtag"

    weights1_path = 'weights1.pkl'
    weights2_path = 'weights2.pkl'
    predictions_path = 'predictions.wtag'

    print("Train model 1")
    feature2id1, pre_trained_weights1, train_time1 = train_model(feature_classes1, train1_path,
                                                                threshold, lam, weights1_path, retrain=False)
    # print("Calc model 1 train accuracy")
    # train_accuracy1 = test_model(feature2id1, pre_trained_weights1, train1_path, predictions_path)[0]

    # print(f"Train 1 accuracy: {train_accuracy1}")
    # print(f"Train 1 time: {train_time1}")

    print("Train model 2")
    feature2id2, pre_trained_weights2, train_time2 = train_model(feature_classes2, train2_path,
                                                                threshold, lam, weights2_path, retrain=False)
    # print("Calc model 2 train accuracy")
    # train_accuracy2 = test_model(feature2id2, pre_trained_weights2, train2_path, predictions_path)[0]
    #
    # print(f"Train 2 accuracy: {train_accuracy2}")
    # print(f"Train 2 time: {train_time2}")

    # test_results = test_model(feature2id1, pre_trained_weights1, test_path, predictions_path,
    #                           return_confusion_matrix=True)
    # test_accuracy, _, conf_mat = test_results
    # print(f"Test 1 accuracy: {test_accuracy}")
    # row_sum = conf_mat.sum(axis=1)
    # diag = conf_mat.apply(lambda row: row[row.name])
    # mistakes = row_sum.sub(diag)
    # top10mistakes_tags = list(mistakes.nlargest(10).index)
    # worst_tags_conf_mat = conf_mat.loc[top10mistakes_tags][top10mistakes_tags]
    # print(worst_tags_conf_mat)

    # print("Cross validate model 2")
    # cross_val_accuracy = cross_validation(feature_classes2, train2_path, threshold, lam, weights2_path,
    #                                       test_path, predictions_path, 5)
    # print(f"Cross validation accuracy: {cross_val_accuracy}")

    tag_all_test(comp1_path, pre_trained_weights1, feature2id1, comp1_pred_path)
    tag_all_test(comp2_path, pre_trained_weights2, feature2id2, comp2_pred_path)


if __name__ == '__main__':
    main()
