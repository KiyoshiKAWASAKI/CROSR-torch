import os
import numpy as np
from scipy.special import softmax




seed = 4
result_dir_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/" \
                  "jhuang24/models/crosr/seed_" + str(seed) + "/features"

# Training output
train_feature_path = result_dir_base + "/train_logits.npy"
train_label_path = result_dir_base + "/train_labels.npy"

train_feature = np.load(train_feature_path)
train_label = np.load(train_label_path)

# Validation output
valid_feature_path = result_dir_base + "/valid_logits.npy"
valid_label_path = result_dir_base + "/valid_labels.npy"

valid_feature = np.load(valid_feature_path)
valid_label = np.load(valid_label_path)

# Test known output
test_known_feature_p0_path = result_dir_base + "/test_known_known_p0_logits.npy"
test_known_feature_p1_path = result_dir_base + "/test_known_known_p1_logits.npy"
test_known_feature_p2_path = result_dir_base + "/test_known_known_p2_logits.npy"
test_known_feature_p3_path = result_dir_base + "/test_known_known_p3_logits.npy"

test_known_feature_p0 = np.load(test_known_feature_p0_path)
test_known_feature_p1 = np.load(test_known_feature_p1_path)
test_known_feature_p2 = np.load(test_known_feature_p2_path)
test_known_feature_p3 = np.load(test_known_feature_p3_path)

test_known_label_p0_path = result_dir_base + "/test_known_known_p0_labels.npy"
test_known_label_p1_path = result_dir_base + "/test_known_known_p1_labels.npy"
test_known_label_p2_path = result_dir_base + "/test_known_known_p2_labels.npy"
test_known_label_p3_path = result_dir_base + "/test_known_known_p3_labels.npy"

test_known_label_p0 = np.load(test_known_label_p0_path)
test_known_label_p1 = np.load(test_known_label_p1_path)
test_known_label_p2 = np.load(test_known_label_p2_path)
test_known_label_p3 = np.load(test_known_label_p3_path)

test_known_feature = np.concatenate((test_known_feature_p0,
                                     test_known_feature_p1,
                                     test_known_feature_p2,
                                     test_known_feature_p3,),
                                    axis=0)

test_known_labels = np.concatenate((test_known_label_p0,
                                    test_known_label_p1,
                                    test_known_label_p2,
                                    test_known_label_p3),
                                   axis=0)

# Test unknown output
test_unknown_feature_path = result_dir_base + "/test_unknown_unknown_logits.npy"
test_unknown_feature = np.load(test_unknown_feature_path)




def calculate_mcc(true_pos,
                  true_neg,
                  false_pos,
                  false_neg):
    """

    :param true_pos:
    :param true_neg:
    :param false_pos:
    :param false_negtive:
    :return:
    """

    return (true_neg*true_pos-false_pos*false_neg)/np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*
                                                           (true_neg+false_pos)*(true_neg+false_neg))




def get_train_valid_results(logits,
                           labels):
    """

    :param original_feature:
    :param aug_feature:
    :param labels:
    :return:
    """
    correct = 0
    wrong = 0


    print("Nb samples: ", logits[0].shape)

    for i in range(len(logits)):
        label = labels[i]
        pred = np.argmax(softmax(logits[i]), axis=0)

        if pred == label:
            correct += 1
        else:
            wrong += 1

    accuracy = float(correct) / float(correct + wrong)
    print("Multi class accuracy: ", accuracy)




def get_test_results(known_feature,
                     known_label,
                     unknown_feature,
                     threshold=0.5):
    """

    :param original_feature:
    :param aug_feature:
    :param labels:
    :return:
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    correct = 0
    wrong = 0

    # Process known samples
    for i in range(len(known_feature)):
        target = known_label[i]

        max_prob = np.max(softmax(known_feature[i], axis=0))
        pred = np.argmax(softmax(known_feature[i]), axis=0)

        if max_prob >= threshold:
            if pred == target:
                correct += 1
            else:
                wrong += 1

            true_positive += 1

        else:
            wrong += 1
            false_negative += 1

    # Process unknown
    for i in range(len(unknown_feature)):
        max_prob = np.max(softmax(unknown_feature[i], axis=0))

        if max_prob < threshold:
            true_negative += 1
        else:
            false_positive += 1

    precision = float(true_positive) / float(true_positive + false_positive)
    recall = float(true_positive) / float(true_positive + false_negative)
    f1 = (2 * precision * recall) / (precision + recall)
    mcc = calculate_mcc(true_pos=float(true_positive),
                        true_neg=float(true_negative),
                        false_pos=float(false_positive),
                        false_neg=float(false_negative))
    unknown_acc = float(true_negative)/float(true_negative+false_positive)
    known_acc = float(correct)/float(correct+wrong)

    print("True positive: ", true_positive)
    print("True negative: ", true_negative)
    print("False postive: ", false_positive)
    print("False negative: ", false_negative)
    print("known accuracy: ", known_acc)
    print("Unknown accuracy: ", unknown_acc)
    print("F-1 score: ", f1)
    print("MCC score: ", mcc)




if __name__ == "__main__":
    print("Seed: ", seed)

    print("*" * 40)
    print("Training results:")
    get_train_valid_results(logits=train_feature,
                           labels=train_label)

    print("*" * 40)
    print("Valid results:")
    get_train_valid_results(logits=valid_feature,
                            labels=valid_label)

    print("Binary results")
    get_test_results(known_feature=test_known_feature,
                     known_label=test_known_labels,
                     unknown_feature=test_unknown_feature)
