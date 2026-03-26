#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Implementation of a pitch modulation information hiding detection
-------------
Based on paper:
	G.729A Pitch Modulation Information Hiding Detection Based on Symbiotic Characteristics
-------------
'''

import time
import os, pickle, csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from tqdm import tqdm

def CNN_pitch(file):
    # P_2(2,2) p
    sample = []
    with open(file, "r") as f:
        for line in f:
            line = [int(i) for i in line.split()]
            sample.append(line)

    p = np.zeros((32,))
    p_joint = np.zeros((1024,))
    # P_1(1,2) q
    q = np.zeros((256,))
    q_joint = np.zeros((256 * 32,))
    try:
        for i in range(len(sample) - 1):
            a = int(sample[i][1])
            b = int(sample[i + 1][1])
            p[a] += 1
            p_joint[a * 32 + b] += 1

            A = int(sample[i][0])
            B = int(sample[i][1])
            q[A] += 1
            q_joint[A * 32 + B] += 1

        i = len(sample) - 1
        A = int(sample[i][0])
        B = int(sample[i][1])
        q[A] += 1
        q_joint[A * 32 + B] += 1

    except IndexError:
        print(f"IndexError in file: {file}")

    p += 1e-5  # avoid divider is 0
    p_repeat = np.repeat(p, 32)
    p_conditional = p_joint / p_repeat

    q += 1e-5  # avoid divider is 0
    q_repeat = np.repeat(q, 32)
    q_conditional = q_joint / q_repeat
    return np.concatenate((q_conditional, p_conditional))


def train_and_test_ccn(positive_data_folder, negative_data_folder, t_positive_data_folder, t_negative_data_folder, result_folder):
    NUM_PCA_FEATURE = 100
    NUM_SAMPLE = 10000
    TEST_NUM_SAMPLE = 2000

    build_model = CNN_pitch
    positive_data_files = [os.path.join(positive_data_folder, path) for path in os.listdir(positive_data_folder)]
    negative_data_files = [os.path.join(negative_data_folder, path) for path in os.listdir(negative_data_folder)]

    t_positive_data_files = [os.path.join(t_positive_data_folder, path) for path in os.listdir(t_positive_data_folder)]
    t_negative_data_files = [os.path.join(t_negative_data_folder, path) for path in os.listdir(t_negative_data_folder)]

    train_positive_data_files = positive_data_files[:NUM_SAMPLE]
    train_negative_data_files = negative_data_files[:NUM_SAMPLE]

    test_positive_data_files = t_positive_data_files[:TEST_NUM_SAMPLE]
    test_negative_data_files = t_negative_data_files[:TEST_NUM_SAMPLE]

    num_train_files = len(train_negative_data_files)
    num_test_files = len(test_negative_data_files)

    # calculate conditional probability matrix
    print("Calculating conditional probability matrix")
    feature = []
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_negative_data_files[i])
        feature.append(new_feature)
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_positive_data_files[i])
        feature.append(new_feature)
    feature = np.row_stack(feature)

    # calculate PCA matrix
    print("Calculating PCA matrix")
    pca = PCA(n_components=NUM_PCA_FEATURE)
    pca.fit(feature)
    with open(os.path.join(result_folder, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    # load train data
    print("Loading train data")
    X = []
    Y = []
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_negative_data_files[i])
        X.append(pca.transform(new_feature.reshape(1, -1)))
        Y.append(0)
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_positive_data_files[i])
        X.append(pca.transform(new_feature.reshape(1, -1)))
        Y.append(1)
    X = np.row_stack(X)

    # train SVM
    print("Training SVM")
    clf = svm.SVC()
    clf.fit(X, Y)
    with open(os.path.join(result_folder, "svm.pkl"), "wb") as f:
        pickle.dump(clf, f)
    
    # test
    print("Testing")
    X = []
    Y = []
    total_inference_time = 0.0

    for i in tqdm(range(num_test_files)):
        start_time = time.time()
        new_feature = build_model(test_negative_data_files[i])
        new_feature = pca.transform(new_feature.reshape(1, -1))
        X.append(new_feature)
        Y.append(0)
        end_time = time.time()
        total_inference_time += (end_time - start_time)

    for i in tqdm(range(num_test_files)):
        start_time = time.time()
        new_feature = build_model(test_positive_data_files[i])
        new_feature = pca.transform(new_feature.reshape(1, -1))
        X.append(new_feature)
        Y.append(1)
        end_time = time.time()
        total_inference_time += (end_time - start_time)

    X = np.row_stack(X)
    Y_predict = clf.predict(X)
    
    with open(os.path.join(result_folder, "Y_predict.pkl"), "wb") as f:
        pickle.dump(Y_predict, f)

    # output result
    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0
    print("Outputing result")
    with open(os.path.join(result_folder, "result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "real class", "predict class"])
        for i in range(num_test_files):
            writer.writerow([test_negative_data_files[i], 0, Y_predict[i]])
            if Y_predict[i] == 0:
                true_negative += 1
            else:
                false_positive += 1

        for i in range(num_test_files):
            writer.writerow([test_positive_data_files[i], 1, Y_predict[i + num_test_files]])
            if Y_predict[i + num_test_files] == 1:
                true_positive += 1
            else:
                false_negative += 1

        writer.writerow(["num of test files", 2 * num_test_files])
        writer.writerow(["True Positive", true_positive])
        writer.writerow(["True Negative", true_negative])
        writer.writerow(["False Positive", false_positive])
        writer.writerow(["False Negative", false_negative])
        accuracy = float(true_negative + true_positive) / (num_test_files * 2) if num_test_files > 0 else 0
        precision = float(true_positive) / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = float(true_positive) / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        avg_inference_time = total_inference_time / (2 * num_test_files) if num_test_files > 0 else 0
        writer.writerow(["Average Inference Time per Sample", avg_inference_time])

    print(f"Average Inference Time per Sample: {avg_inference_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}") 