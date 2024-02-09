import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from check_submission import compare_files


def main():
    threshold = 1
    lam = 1

    train_path = "../data/train1.wtag"
    test_path = "../data/test1.wtag"

    weights_path = '../weights.pkl'
    predictions_path = '../predictions.wtag'

    # statistics, feature2id = preprocess_train(train_path, threshold)
    # get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    # tag_all_test(train_path, pre_trained_weights, feature2id, predictions_path)
    # print(compare_files(train_path, predictions_path))

    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    print(compare_files(test_path, predictions_path))


if __name__ == '__main__':
    main()
