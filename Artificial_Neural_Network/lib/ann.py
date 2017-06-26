#!/usr/bin/env python3
# coding=utf-8
# -*- utf8 -*-
# author=dave.fang@outlook.com
# create=20170427
import math
import random

import pandas
import numpy as np
import matplotlib.pyplot as plt
from lib.log import LOGGER


class Ann:
    def __init__(self, input_layer, hidden_layer, output_layer, test_ratio=0.2):
        self.iris_data = []
        self.iris_species = []
        self.test_ratio = test_ratio

        self.costs = []
        self.correct = 0.1
        self.learning_rate = 0.03

        # 定义输入层、隐藏层、输出层的层数
        self.input_layer = input_layer + 1
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        # 初始化节点
        self.input_cells = [1.0] * self.input_layer
        self.hidden_cells = [1.0] * self.hidden_layer
        self.output_cells = [1.0] * self.output_layer

        # 初始化权重
        self.input_weights = make_matrix(self.input_layer, self.hidden_layer)
        self.output_weights = make_matrix(self.hidden_layer, self.output_layer)

        # 随机初始权重
        for i in range(self.input_layer):
            for h in range(self.hidden_layer):
                self.input_weights[i][h] = random.uniform(-2.0, 2.0)
        for h in range(self.hidden_layer):
            for o in range(self.output_layer):
                self.output_weights[h][o] = random.uniform(-2.0, 2.0)

        # 初始化修正集合
        self.input_correction = make_matrix(self.input_layer, self.hidden_layer)
        self.output_correction = make_matrix(self.hidden_layer, self.output_layer)

        # 训练集 与 测试集 的初始化
        self.train_feature = []
        self.train_specie = []
        self.test_feature = []
        self.test_specie = []

        # 置随机种子
        random.seed(0)

    def import_data(self, file_path):
        """
        导入数据
        :param file_path:
        :return:
        """
        LOGGER.info('样例数据导入路径: {0}'.format(file_path))

        self.iris_data = pandas.read_csv(file_path)
        shuffled_rows = np.random.permutation(self.iris_data.index)
        # 打乱输入样本数据的顺序
        self.iris_data = self.iris_data.loc[shuffled_rows, :]
        self.iris_species = self.iris_data.species.unique()

        LOGGER.info('鸢尾花类别: {0}'.format(self.iris_species))

        # print(self.iris_data.head())

    def pre_data(self):
        """
        样本数据预处理
        :return:
        """
        num_train = round(len(self.iris_data) * self.test_ratio)
        LOGGER.info('测试集占比为 {0}% , 测试集数量为 {1}'.format(self.test_ratio * 100, num_train))

        # 样本属性标准化
        feature_data = self.iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        feature_data = normalize(feature_data)

        # 样本种类标签化,即用数组表示种类
        specie_data = self.iris_data.species.values
        for i in range(len(self.iris_species)):
            label = [0 for x in range(len(self.iris_species))]
            label[i] = 1
            LOGGER.info('{0}\t=>\t{1}'.format(self.iris_species[i], label))
            specie_data = [label if x == self.iris_species[i] else x for x in specie_data]

        # 分配训练集数据
        self.train_feature = feature_data[num_train:]
        self.train_specie = specie_data[num_train:]

        # 分配测试集数据
        self.test_feature = feature_data[0:num_train]
        self.test_specie = specie_data[0:num_train]

    def train(self, data, labels, max_epochs=10000, learning_rate=0.05, correct=0.1):
        """
        神经网络样本训练
        :param data:
        :param labels:
        :param max_epochs:
        :param learning_rate:
        :param correct:
        :return:
        """
        self.costs = []
        self.correct = correct
        self.learning_rate = learning_rate
        for i in range(int(max_epochs)):

            error = 0.0
            for j in range(len(data)):
                label = labels[j]
                case = data[j]
                error += self.back_propagate(case, label)
            self.costs.append(error)
            if i % 100 == 0:
                LOGGER.info('第 {0} 次数据迭代, 错误率为 {1}'.format(i, error))

    def predict(self, inputs):
        """
        预测函数
        :param inputs:
        :return:
        """
        # activate input layer
        for i in range(self.input_layer - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_layer):
            total = 0.0
            for i in range(self.input_layer):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_layer):
            total = 0.0
            for j in range(self.hidden_layer):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, data, label):
        """
        反向传播,从最后一层绕回到最开始的一层来逐层修改参数
        :param data:
        :param label:
        :return:
        """
        # 前向传播
        self.predict(data)
        # 获取输出层的误差
        output_deltas = [0.0] * self.output_layer
        for o in range(self.output_layer):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # 获取隐藏层的误差
        hidden_deltas = [0.0] * self.hidden_layer
        for h in range(self.hidden_layer):
            error = 0.0
            for o in range(self.output_layer):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # 更新输出层的权重
        for h in range(self.hidden_layer):
            for o in range(self.output_layer):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += self.learning_rate * change + self.correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # 更新输入层的权重
        for i in range(self.input_layer):
            for h in range(self.hidden_layer):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += self.learning_rate * change + self.correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 获取全局误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def test(self):
        self.train(self.train_feature, self.train_specie)
        test_wrong = 0
        for i in range(len(self.test_feature)):
            predict_data = self.predict(self.test_feature[i])
            for j in range(len(predict_data)):
                predict_data[j] = round(predict_data[j])
            if predict_data == self.test_specie[i]:
                LOGGER.info('[CORRECT] {0} / {1}'.format(predict_data, self.test_specie[i]))
                pass
            else:
                LOGGER.info('[WRONG] {0} / {1}'.format(predict_data, self.test_specie[i]))
                test_wrong += 1

        accuracy = 1 - float(test_wrong / round(self.test_ratio * len(self.iris_data)))
        LOGGER.info('测试集准确度为 {0}%'.format(accuracy*100))

        plt.plot(self.costs)
        plt.title("Error of each Iteration")
        plt.ylabel("Error")
        plt.xlabel("Iteration")
        plt.show()


def normalize(param):
    """
    样本预处理,标准化数据
    :param param:
    :return:
    """
    param_max, param_min = param.max(), param.min()
    return (param - param_min) / (param_max - param_min)


def sigmoid(x):
    """
    神经网络激活函数
    :param x:
    :return:
    """
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    """
    神经网络激活函数的导数
    :param x:
    :return:
    """
    return x * (1 - x)


def make_matrix(m, n, fill=0.0):
    """
    生成大小 m*n 的矩阵，默认零矩阵
    :param m:
    :param n:
    :param fill:
    :return:
    """
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat
