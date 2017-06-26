#!/usr/bin/env python3
# coding=utf-8
# -*- utf8 -*-
# author=dave.fang@outlook.com
# create=20170516
from lib.ga import GA
from lib.log import LOGGER


def main():
    # min[3-sin(jx1)^2-sin(jx2)^2]
    # 实例化遗传算法类
    ga = GA(1000, 100, 0, 6)
    ga.start()

if __name__ == '__main__':
    LOGGER.info('遗传算法 HW3')
    main()
