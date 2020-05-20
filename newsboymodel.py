# coding=utf-8
'''
:func 报童模型演算
:author: MrLi
:date: 

'''
import sys
import os
import codecs
import json
import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def normal(x, u, v):
    return 1 / (v * np.sqrt(2 * np.pi)) * np.exp(-(x - u) ** 2 / 2 / v**2)


def main():
    u = 15
    v = 3
    accu = 0.01
    x = np.arange(u - 1 * u, u + 1 * u, accu)
    y = normal(x, u, v)
    # y = y / max(y)
    
    # 利润
    price = 22
    cost = 15
    """分情况考虑
    1. 进货量<=需求量-(price-cost)*x
    2. 进货量>需求量-(price*y-cost)*x
    """
    
    def profit(c, uc):
        if c <= uc:
            return c * (price - cost)
        else:
            return uc * price - cost * c
    
    # 成本
    costline = x * cost
    # 利润
    # profitline = list()
    
    # 当进货量为c
    # for c in x:
    #     # 第二天销售量期望
    #     profitline.append(sum(np.array([profit(c, uc) for uc in x])*y))
    profitline = [sum(np.array([profit(c, uc) for uc in x]) * y)
                  for c in x]
    
    plt.plot(x, y / max(y), label='正态分布')
    plt.plot(x, costline / max(costline), label='成本')
    plt.plot(x, profitline / max(profitline), label='利润')
    
    max_indx = np.argmax(profitline)
    print(max_indx, y[max_indx] / max(y))
    show_max = '({x},{y})'.format(x=round(max_indx * accu, 2), y=round(profitline[max_indx] * accu, 2))
    plt.annotate(show_max, xy=(max_indx * accu, profitline[max_indx] / max(profitline)))
    plt.scatter(max_indx * accu, profitline[max_indx] / max(profitline))
    plt.title('顾客数量与利润,成本分布图')
    plt.xlabel('顾客数量')
    plt.ylabel('标准化')
    
    # 由于订货量为整数
    c = max_indx * accu
    for i, c in enumerate((int(c), int(c)+1)):
        profit_max = sum(np.array([profit(c, uc)
                                   for uc in x] * y))*accu
        plt.annotate('{c}-{profit}'.format(c=c, profit=round(profit_max, 2)),
                     xy=(3, -0.75+i*0.25))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
