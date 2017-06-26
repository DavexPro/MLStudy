#!/usr/bin/env python3
# coding=utf-8
# -*- utf8 -*-
# author=dave.fang@outlook.com
# create=20170516
import copy
import math

import numpy as np
import matplotlib.pyplot as plt

from lib.log import LOGGER


class GA:
    def __init__(self, epochs, pop_size, range_min, range_max, cross_rate=0.75, mutate_rate=0.05):
        """
        初始化参数
        :param epochs: 进化次数
        :param pop_size: 种群大小
        :param range_min: 函数区间最小值
        :param range_max: 函数区间最大值
        :param cross_rate: 交叉概率
        :param mutate_rate: 突变概率
        """
        # 遗传进化代数
        self.epochs = epochs
        self.params_num = 1

        # 种群每代的个数
        self.pop_size = pop_size
        self.chromosome_size = 0
        self.chromosome_length = 1

        # 函数输入区间
        self.range_min = range_min
        self.range_max = range_max

        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate

        # 每代最优个体
        self.pop_best = []
        self.population = []
        self.chromosome_dec = []

    def init_population(self):
        """
        初始化种群
        :return:
        """
        self.population = np.random.randint(0, 2, size=(self.pop_size, self.chromosome_size))

    def init_chromosome_size(self, accuracy, params_num=1):
        """
        根据求解精度和参数个数得出编码长度
        :param accuracy: 计算精度
        :param params_num: 参数个数
        :return:
        """
        self.params_num = params_num
        size = (self.range_max - self.range_min) * 10 ** accuracy
        chromosome_size = 1
        while size / 2 > 1:
            size /= 2
            chromosome_size += 1
        self.chromosome_length = chromosome_size
        self.chromosome_size = chromosome_size * self.params_num
        LOGGER.info('编码的二进制串长度为: {0} / 单位长度为: {1}'.format(self.chromosome_size, self.chromosome_length))

    def bin2dec(self, chromosome):
        """
        将代表染色体的二进制数组转化为十进制
        :param chromosome:
        :return:
        """
        result = 0
        for gene in range(self.chromosome_length):
            result += chromosome[gene] * 2 ** gene
        return result

    def get_dec_list(self):
        """
        将十进制染色体转换到求解空间中的数值
        :return:
        """
        self.chromosome_dec = []
        step = (self.range_max - self.range_min) / float(2 ** self.chromosome_length - 1)
        for pop in range(self.pop_size):
            arr_param = []
            for i in range(self.params_num):
                origin_chromosome = self.population[pop][self.chromosome_length * i:self.chromosome_length * (i + 1)]
                decimal = self.range_min + step * self.bin2dec(origin_chromosome)
                arr_param.append(decimal)
            self.chromosome_dec.append(arr_param)
        return self.chromosome_dec

    @staticmethod
    def evaluate_func(param):
        """
        计算函数值 3-sin(jx1)^2-sin(jx2)^2
        :param param:
        :return:
        """
        constant_j = 2
        return math.sin(constant_j * param[0]) ** 2 + math.sin(constant_j * param[1]) ** 2

    def get_fitness(self, params):
        """
        对原函数进行变形后, 等价于求新函数的最大值, 因此在此的适应值为新的函数值
        :param params:
        :return:
        """
        fitness = []
        for param in params:
            fitness.append(self.evaluate_func(param))
        return fitness

    def selection(self, fit_value):
        """
        通过轮盘赌方法进行选择
        :param fit_value:
        :return:
        """
        new_fit = []
        total_fit = sum(fit_value)
        accumulator = 0.0
        for val in fit_value:
            # 对每一个适应度除以总适应度，然后累加，这样可以使适应度大的个体获得更大的比例空间。
            new_val = (val * 1.0 / total_fit)
            accumulator += new_val
            new_fit.append(accumulator)

        ms = []
        for i in range(self.pop_size):
            # 随机生成0,1之间的随机数
            ms.append(np.random.random())

        # 对随机数进行排序
        ms.sort()

        index_fit = 0
        index_new = 0
        population_new = self.population
        while index_new < self.pop_size:
            # 随机投掷，选择落入个体所占轮盘空间的个体
            if ms[index_new] < new_fit[index_fit]:
                population_new[index_new] = self.population[index_fit]
                index_new += 1
            else:
                index_fit += 1

        # 适应度大的个体会被选择的概率较大
        # 使得新种群中，会有重复的较优个体
        self.population = population_new

    def crossover(self):
        """
        单点杂交, 采取近邻个体杂交
        :return:
        """
        for i in range(self.pop_size - 1):
            # 随机数小于杂交率, 则进行杂交操作
            if np.random.random() < self.cross_rate:
                # 随机选择交叉点
                single_point = np.random.randint(0, self.chromosome_size)

                # 对两条染色体进行切片、重组
                tmp_1 = []
                tmp_1.extend(self.population[i][:single_point])
                tmp_1.extend(self.population[i + 1][single_point:])

                tmp_2 = []
                tmp_2.extend(self.population[i + 1][:single_point])
                tmp_2.extend(self.population[i][single_point:])

                self.population[i], self.population[i + 1] = tmp_1, tmp_2

    def mutation(self):
        """
        变异
        :return:
        """
        for i in range(self.pop_size):
            # 随机数小于变异率，则进行变异操作
            if np.random.random() < self.mutate_rate:
                mutate_point = np.random.randint(0, self.chromosome_size - 1)
                # 将随机点上的基因进行反转
                if self.population[i][mutate_point] == 1:
                    self.population[i][mutate_point] = 0
                else:
                    self.population[i][mutate_point] = 1

    def elitism(self, pop_best, after_fit_best, fit_best, after_fit_list):
        """
        精英主义选择, 剔除差的染色体, 替换成适应度最强的染色体
        :param pop_best:
        :param after_fit_best:
        :param fit_best:
        :param after_fit_list:
        :return:
        """
        # 上一代的最优适应度，本代最优适应度。这些变量是在主函数中生成的。
        if after_fit_best - fit_best < 0:
            # 满足精英策略后，找到最差个体的索引，进行替换。
            pop_worst = after_fit_list.index(min(after_fit_list))
            self.population[pop_worst] = pop_best

    def start(self):
        # 计算染色体编码长度
        self.init_chromosome_size(6, 2)
        # 种群初始化
        self.init_population()

        # 在遗传代数内进行迭代
        for g in range(self.epochs):
            dec_list = self.get_dec_list()
            fit_list = self.get_fitness(dec_list)  # 适应度函数
            pop_best = self.population[fit_list.index(max(fit_list))]
            fit_best = max(fit_list)
            self.pop_best.append(fit_best)

            # 对 pop_best 进行深复制，以为后面精英选择做准备
            pop_best = copy.deepcopy(pop_best)

            self.selection(fit_list)  # 选择
            self.crossover()  # 交叉
            self.mutation()  # 变异

            # 对变异之后的种群，求解最大适应度
            after_dec_list = self.get_dec_list()
            after_fit_list = self.get_fitness(after_dec_list)
            after_fit_best = max(after_fit_list)

            LOGGER.info('第 {0} 代: 适应度 {1} / 变异后适应度 {2}'.format((g + 1), fit_best, after_fit_best))

            self.elitism(pop_best, after_fit_best, fit_best, after_fit_list)

        axis_x = [x for x in range(self.epochs)]
        axis_y = self.pop_best

        LOGGER.info('遗传进化结束, 最终适应度为 {0} , 对应函数计算结果 {1}'.format(fit_best, (3 - fit_best)))

        plt.plot(axis_x, axis_y)
        plt.title('Pc={0}, Pm={1}'.format(self.cross_rate, self.mutate_rate))
        plt.show()
        plt.close()


if __name__ == '__main__':
    ga = GA(200, 50, 0, 6)
    ga.start()
