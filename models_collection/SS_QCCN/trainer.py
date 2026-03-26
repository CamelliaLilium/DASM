#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, pickle, csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from tqdm import tqdm
import time

def G729_SS_QCCCN(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) < 3:
                continue
            try:
                line = [int(i) for i in line[0:3]]
            except ValueError:
                continue
            data.append(line)

    a = np.zeros(shape=(128, 128))
    c1 = np.zeros(shape=128)
    p = np.zeros(shape=(32, 32))
    c2 = np.zeros(shape=32)

    for i in range(len(data) - 1):
        data1 = data[i]
        data2 = data[i + 1]
        c1[data1[0]] += 1
        c2[data1[1]] += 1
        a[data1[0], data2[0]] += 1
        p[data1[1], data1[2]] += 1

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if c1[i] != 0:
                a[i, j] /= c1[i]

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if c2[i] != 0:
                p[i, j] /= c2[i]

    return np.concatenate([a.reshape(128 * 128), p.reshape(32 * 32)])

def train_and_test_ss_qccn(positive_data_folder, negative_data_folder, t_positive_data_folder, t_negative_data_folder, result_folder):
    NUM_SAMPLE = 10000
    TEST_NUM_SAMPLE = 2000
    NUM_PCA_FEATURE = 300
    
    build_model = G729_SS_QCCCN

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

    print("Calculating PCA matrix")
    feature = []
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_negative_data_files[i])
        feature.append(new_feature)
    for i in tqdm(range(num_train_files)):
        new_feature = build_model(train_positive_data_files[i])
        feature.append(new_feature)
    feature = np.row_stack(feature)

    n_components = min(NUM_PCA_FEATURE, min(num_train_files * 2, feature.shape[1]))
    pca = PCA(n_components=n_components)
    pca.fit(feature)

    with open(os.path.join(result_folder, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

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

    print("Training SVM")
    clf = svm.SVC()
    clf.fit(X, Y)
    with open(os.path.join(result_folder, "svm.pkl"), "wb") as f:
        pickle.dump(clf, f)
    
    print("Testing")
    X = []
    total_inference_time = 0.0

    for i in tqdm(range(num_test_files)):
        start_time = time.time()
        new_feature = build_model(test_negative_data_files[i])
        new_feature = pca.transform(new_feature.reshape(1, -1))
        X.append(new_feature)
        end_time = time.time()
        total_inference_time += (end_time - start_time)

    for i in tqdm(range(num_test_files)):
        start_time = time.time()
        new_feature = build_model(test_positive_data_files[i])
        new_feature = pca.transform(new_feature.reshape(1, -1))
        X.append(new_feature)
        end_time = time.time()
        total_inference_time += (end_time - start_time)

    X = np.row_stack(X)
    Y_predict = clf.predict(X)
    
    with open(os.path.join(result_folder, "Y_predict.pkl"), "wb") as f:
        pickle.dump(Y_predict, f)

    true_negative = np.sum(Y_predict[:num_test_files] == 0)
    false_positive = np.sum(Y_predict[:num_test_files] == 1)
    true_positive = np.sum(Y_predict[num_test_files:] == 1)
    false_negative = np.sum(Y_predict[num_test_files:] == 0)

    print("Outputing result")
    with open(os.path.join(result_folder, "result.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "real class", "predict class"])
        # Writing test results can be added here if needed

        writer.writerow(["num of test files", 2 * num_test_files])
        writer.writerow(["True Positive", true_positive])
        writer.writerow(["True Negative", true_negative])
        writer.writerow(["False Positive", false_positive])
        writer.writerow(["False Negative", false_negative])
        accuracy = (true_positive + true_negative) / (num_test_files * 2) if num_test_files > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        avg_inference_time = total_inference_time / (2 * num_test_files) if num_test_files > 0 else 0
        writer.writerow(["Average Inference Time per Sample", avg_inference_time])

    print(f"Accuracy: {accuracy:.4f}") 