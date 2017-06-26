#!/usr/bin/env python3
# coding=utf-8
# -*- utf8 -*-
# author=dave.fang@outlook.com
# create=20170330
from lib.log import LOGGER
from lib.naive_bayes import NaiveBayes


def main():
    # 实例化朴素贝叶斯操作类
    bayes = NaiveBayes()
    bayes.import_data('./input.txt')
    bayes.separate_data()
    bayes.analyse_data()
    # 人身高6英尺、体重130磅，脚掌8英寸
    bayes.classify(6, 130, 8)


if __name__ == '__main__':
    LOGGER.info('朴素贝叶斯实验 HW1')
    main()
