import numpy as np

from preprocessing import read_test
from tqdm import tqdm
from preprocessing import represent_input_with_features


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    n = len(sentence)
    beam = 3
    best_past_tags = {("", ""): (1, ["", ""])}
    all_tags = feature2id.feature_statistics.tags
    # calc best route
    for k in range(2, n):  # scan over all histories
        c_pi = {}  # current probability matrix
        for pp_tag, p_tag in best_past_tags:  # scan for every beam
            feature_vectors = {}  # the feature vectors of all history combinations
            for c_tag in all_tags:
                hist_ = history(k, pp_tag, p_tag, c_tag)
                feature_vector = np.array(represent_input_with_features(hist_, feature2id.feature_to_idx))
                feature_vectors[c_tag] = feature_vector
            # the denominator for calculating c_pi

            denominator = sum([np.exp(np.dot(pre_trained_weights, feature_vectors[c_tag])) for c_tag in all_tags])

            for c_tag in all_tags:  # find the best one-step routs from all beams
                feature_vector = feature_vectors[c_tag]
                soft_max_ = soft_max(pre_trained_weights, feature_vector, denominator)

                curr_prob = soft_max_ * best_past_tags[(pp_tag, p_tag)][0]
                if not c_pi[(p_tag, c_tag)] or curr_prob > c_pi[(p_tag, c_tag)]:
                    c_pi[(p_tag, c_tag)] = [curr_prob, best_past_tags[(pp_tag, p_tag)][1] + [c_tag]]
        # save {beam} best routs
        best_past_tags = {}
        for pp_tag, p_tag in sorted(c_pi, key=c_pi.get, reverse=False)[:beam]:
            best_past_tags[(pp_tag, p_tag)] = c_pi[(pp_tag, p_tag)]

    # pick best route
    return max(best_past_tags)[1]


def history(sentence, k, pp_tag, p_tag, c_tag):
    return sentence[k], c_tag, sentence[k - 1], p_tag, sentence[k - 2], pp_tag, sentence[k + 1]


def soft_max(weights, feature_vector, denominator):
    return np.exp(weights * feature_vector) / denominator


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()