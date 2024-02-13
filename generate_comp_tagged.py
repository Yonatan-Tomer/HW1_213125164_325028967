from inference import tag_all_test
from main import train_model


def main():
    """
    Tag competitions
    """
    feature_classes1 = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "capital", "number",
                       "contains hyphen", "pp_word", "f100_lower", "f101_lower", "f102_lower", "f106_lower",
                       "f107_lower", "nn_word", "c_word_n_word", "length", "2words", "3words"]

    feature_classes2 = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "capital", "number",
                       "contains hyphen", "pp_word", "length"]
    threshold = 1
    lam = 1

    train1_path = "data/train1.wtag"
    train2_path = "data/train2.wtag"
    comp1_path = "data/comp1.words"
    comp2_path ="data/comp2.words"
    comp1_pred_path = "comp_m1_213125164_325028967.wtag"
    comp2_pred_path = "comp_m2_213125164_325028967.wtag"

    weights1_path = 'weights1.pkl'
    weights2_path = 'weights2.pkl'

    # load model 1
    feature2id1, pre_trained_weights1, train_time1 = train_model(feature_classes1, train1_path,
                                                                threshold, lam, weights1_path, retrain=False)

    # load model 2
    feature2id2, pre_trained_weights2, train_time2 = train_model(feature_classes2, train2_path,
                                                                threshold, lam, weights2_path, retrain=False)

    # tag
    tag_all_test(comp1_path, pre_trained_weights1, feature2id1, comp1_pred_path)
    tag_all_test(comp2_path, pre_trained_weights2, feature2id2, comp2_pred_path)


if __name__ == '__main__':
    main()