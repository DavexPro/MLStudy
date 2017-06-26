#!/usr/bin/env python3
# coding=utf-8
# -*- utf8 -*-
# author=dave.fang@outlook.com
# create=20170527

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.log import LOGGER
from lib.kmeans import KMeans


def main():
    num_k = 3
    data_x = import_data('./input.txt')
    exam_x, exam_y = get_exam_data('./input.txt')

    k_mean = KMeans(3, 1000)
    # 开始训练模型
    k_mean.fit(data_x)
    result = k_mean.predict(exam_x)

    exam_accuracy(result, exam_y)

    cents = k_mean.centroids
    labels = k_mean.labels
    sse = k_mean.sse
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']

    for i in range(num_k):
        index = np.nonzero(labels == i)[0]
        x0 = data_x[index, 0]
        x1 = data_x[index, 1]
        y_i = i
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(y_i), color=colors[i], fontdict={'weight': 'bold', 'size': 6})
        plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i], linewidths=7)

    plt.title("SSE={:.2f}".format(sse))
    plt.axis([3, 9, 0, 6])
    filename = "./result/k_clusters" + str(num_k) + ".png"
    plt.savefig(filename)
    plt.show()


def exam_accuracy(result, labels):
    clusters = {
        '0': [],
        '1': [],
        '2': [],
    }

    for i in range(len(result)):
        cluster_id = '{0}'.format(int(result[i]))
        clusters[cluster_id].append(labels[i][0])

    clusters_0 = set(clusters['0'])
    clusters_1 = set(clusters['1'])
    clusters_2 = set(clusters['2'])

    num_wrong = 0

    max_num = 0
    for item in clusters_0:
        if max_num < clusters['0'].count(item):
            max_num = clusters['0'].count(item)

    num_wrong += len(clusters['0']) - max_num

    max_num = 0
    for item in clusters_1:
        if max_num < clusters['1'].count(item):
            max_num = clusters['1'].count(item)

    num_wrong += len(clusters['1']) - max_num

    max_num = 0
    for item in clusters_2:
        if max_num < clusters['2'].count(item):
            max_num = clusters['2'].count(item)

    num_wrong += len(clusters['2']) - max_num

    LOGGER.info('{0} Wrongs Found, Accuracy: {1}%'.format(num_wrong, ((len(result) - num_wrong) / len(result)) * 100))


def get_exam_data(filename):
    """
    获取用于校验的数据
    :param filename:
    :return:
    """
    iris_data = pd.read_csv(filename, sep=',', header=0)

    shuffled_rows = np.random.permutation(iris_data.index)

    # 打乱输入样本数据的顺序
    iris_data = iris_data.loc[shuffled_rows, :]

    iris_x = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    iris_x = np.float32(iris_x)

    iris_y = iris_data[['species']].values

    return iris_x, iris_y


def import_data(filename):
    """
    导入数据
    :param filename:
    :return:
    """
    LOGGER.info('样例数据导入路径: {0}'.format(filename))

    iris_data = pd.read_csv(filename, sep=',', header=0)

    shuffled_rows = np.random.permutation(iris_data.index)

    # 打乱输入样本数据的顺序
    iris_data = iris_data.loc[shuffled_rows, :]
    iris_species = iris_data.species.unique()

    iris_x = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    iris_x = np.float32(iris_x)

    LOGGER.info('鸢尾花类别: {0}'.format(iris_species))
    return iris_x


if __name__ == "__main__":
    LOGGER.info('k-Means HW4')
    main()
