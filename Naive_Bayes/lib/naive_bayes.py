#!/usr/bin/env python3
# coding=utf-8
# -*- utf8 -*-
# author=dave.fang@outlook.com
# create=20170330
import math
from lib.log import LOGGER


class NaiveBayes:
    def __init__(self):
        # 元数据
        self.data_set = []
        # 分类后的数据
        self.separate_set = {}
        # 分类后, 性别的先验概率
        self.separate_prior = {}
        # 分类数据的参数分类
        self.separate_probability = {}

    def import_data(self, file_path):
        """
        导入元数据
        :param file_path:
        :return:
        """
        LOGGER.info('样例数据导入路径: {0}'.format(file_path))

        file_handle = open(file_path)
        file_content = file_handle.read().split('\n')
        for line in file_content:
            if line.strip() == '' or line[0] == '#':
                continue
            tmp_set = line.split(' ')
            self.data_set.append(tmp_set)

        LOGGER.info('样例数据集导入完成...')

    def separate_data(self):
        """
        分类样例数据, 并计算性别的先验概率
        :return:
        """
        LOGGER.info('开始分类样例数据...')

        # 根据性别对样例数据进行分类
        for one_set in self.data_set:
            if one_set[0] not in self.separate_set:
                self.separate_set[one_set[0]] = []
                self.separate_probability[one_set[0]] = []
                self.separate_prior[one_set[0]] = 0
            self.separate_set[one_set[0]].append(one_set)

        # 计算性别的先验概率 P(gender)
        for one_prior in self.separate_prior:
            self.separate_prior[one_prior] = len(self.separate_set[one_prior]) / len(self.data_set)

        LOGGER.info('样例数据分类完成...')

    def analyse_data(self):
        """
        对分类数据进行再次加工, 方便后续的取均值以及标准差的计算
        :return:
        """
        for one_separate in self.separate_set:
            self.separate_probability[one_separate] = {
                'height': [],
                'weight': [],
                'foot': []
            }
            for one_set in self.separate_set[one_separate]:
                self.separate_probability[one_separate]['height'].append(float(one_set[1]))
                self.separate_probability[one_separate]['weight'].append(float(one_set[2]))
                self.separate_probability[one_separate]['foot'].append(float(one_set[3]))

    def classify(self, height, weight, foot):
        """
        根据所给数据进行分类, 并给出判断
        :param height:
        :param weight:
        :param foot:
        :return: 性别
        """
        LOGGER.info('数据分类: 身高{0}英尺 / 体重{1}磅 / 脚掌{2}英寸'.format(height, weight, foot))
        category = {}
        for one_separate in self.separate_set:
            # 计算所给身高在该类别(性别)的概率密度
            pro_height = calc_probability(height, calc_mean(self.separate_probability[one_separate]['height']),
                                          calc_stdev(self.separate_probability[one_separate]['height']))

            # 计算所给体重在该类别(性别)的概率密度
            pro_weight = calc_probability(weight, calc_mean(self.separate_probability[one_separate]['weight']),
                                          calc_stdev(self.separate_probability[one_separate]['weight']))

            # 计算所给脚长在该类别(性别)的概率密度
            pro_foot = calc_probability(foot, calc_mean(self.separate_probability[one_separate]['foot']),
                                        calc_stdev(self.separate_probability[one_separate]['foot']))

            category[one_separate] = self.separate_prior[one_separate] * pro_height * pro_weight * pro_foot

        # 两个概率比较, 取最大值作为我们最后分类的结果
        if category['0'] / category['1'] > 1:
            LOGGER.info('女性 / 女性的概率比男性的概率高{0}倍'.format(round(category['0'] / category['1'])))
        else:
            LOGGER.info('男性 / 男性的概率比女性的概率高{0}倍'.format(round(category['1'] / category['0'])))


def calc_mean(numbers):
    """
    计算一组数的均值
    :param numbers:
    :return: 均值
    """
    return sum(numbers) / float(len(numbers))


def calc_stdev(numbers):
    """
    计算一组数的标准差
    :param numbers:
    :return: 标准差
    """
    avg = calc_mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def calc_probability(x, mean, stdev):
    """
    计算概率密度函数的值
    :param x:
    :param mean:
    :param stdev:
    :return:
    """
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
