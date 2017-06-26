#!/usr/bin/env python3
# coding=utf-8
# -*- utf8 -*-
# author=dave.fang@outlook.com
# create=20170427
from lib.ann import Ann
from lib.log import LOGGER


def main():
    # 实例化人工神经网络操作类
    ann = Ann(4, 3, 3)
    ann.import_data('./input.txt')
    ann.pre_data()
    ann.test()


if __name__ == '__main__':
    LOGGER.info('人工神经网络实验 HW2')
    main()
