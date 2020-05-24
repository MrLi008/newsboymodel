# coding=utf-8
'''
:func 回购合同利润计算
回购价越高-销售厂商损失越少
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
    return 1 / (v * np.sqrt(2 * np.pi)) * np.exp(-(x - u) ** 2 / 2 / v ** 2)


def newboys(u, v, accu, price, cost, price_buy_back=0, **kwargs):
    """
    报童模型模拟计算
    总成本: u * cost
    总销售额: u * price
    总利润: uc*price-u*cost [u 订货量,uc 实际销售量 cost 成本价 price 售价]

    :param u: 均值/源数据均值-订货量/销售量-
    :type u: float
    :param v: 标准差-计算正态分布函数
    :type v: float
    :param accu: 精度
    :type accu: 通常为 u/100 u/1000
    :param price: 售价/销售额
    :type price: float
    :param cost: 成本价/供应商给的价格
    :type cost: float
    :param price_buy_back: 回购价格 =0 为不回购
    :type price_buy_back: float
    :return: 利润最大的最优订货量
    :rtype: tuple (订货量,最优利润)
    """
    # u = 15
    # v = 3
    # accu = 0.01
    x = np.arange(u - 1 * u, u + 1 * u, accu)
    y = normal(x, u, v)
    # y = y / max(y)
    
    # 利润
    # price = 22
    # cost = 15
    """分情况考虑
    1. 进货量<=需求量-(price-cost)*x
    2. 进货量>需求量-(price*y-cost)*x
    """
    
    def profit(c, uc):
        """
        利润计算
        :param c: 进货量
        :type c: float
        :param uc: 销售量
        :type uc: float
        :return: 对应利润
        :rtype: float
        """
        if c <= uc:
            # 利润 = (售价-成本)*销售量
            return c * (price - cost)
        else:
            # 对销售商来讲 利润 = (售价-成本)*销售量+(回购价-成本)*未售出量
            return uc * (price - cost) + (price_buy_back - cost) * (c - uc)
            # 对供应商来讲 利润 = (售价-成本)*销售量-(回购价+成本)*未售出量
            # return uc * (price - cost) - (price_buy_back + cost) * (c - uc)
    
    # 成本
    costline = x * cost
    # 利润
    # profitline = list()
    
    # 当进货量为c
    # for c in x:
    #     # 第二天销售量期望
    #     profitline.append(sum(np.array([profit(c, uc) for uc in x])*y))
    profitline = [np.sum(np.array([profit(c, uc) for uc in x]) * y)
                  for c in x]
    
    plt.plot(x, y / max(y), label='正态分布-销售量')
    plt.plot(x, costline / max(costline), label='成本-固定成本')
    plt.plot(x, profitline / max(profitline), label='利润')
    
    max_indx = np.argmax(profitline)
    print('当前 售价: ',price, '成本: ', cost, '进货量: ', max_indx, '利润为: ', profitline[max_indx] *accu)
    show_max = '({x},{y})'.format(x=round(max_indx * accu, 2), y=round(profitline[max_indx] * accu, 2))
    plt.annotate(show_max, xy=(max_indx * accu, profitline[max_indx] / max(profitline)))
    plt.scatter(max_indx * accu, profitline[max_indx] / max(profitline))
    title = kwargs.get('title', '顾客数量与利润,成本分布图')
    plt.title(title)
    plt.xlabel('顾客数量')
    plt.ylabel('标准化')
    
    # 由于订货量为整数
    c = max_indx * accu
    for i, c in enumerate((int(c), int(c) + 1)):
        profit_max = sum(np.array([profit(c, uc)
                                   for uc in x] * y)) * accu
        plt.annotate('{c}-{profit}'.format(c=c, profit=round(profit_max, 2)),
                     xy=(u / 10, 0.75 + i * 0.25))
    plt.legend()
    plt.show()
    
    leftc = int(c - 0.5)
    lprefit = sum(np.array([profit(leftc, uc)
                            for uc in x] * y)) * accu
    rightc = int(c + 0.5)
    rprefit = sum(np.array([profit(rightc, uc)
                            for uc in x] * y)) * accu
    if lprefit < rprefit:
        return rightc, rprefit
    else:
        return leftc, lprefit
    # return [
    #     (c, profitline[max_indx]),
    #     (leftc, sum(np.array([profit(leftc, uc)
    #                                for uc in x] * y)) * accu),
    #     (rightc, sum(np.array([profit(rightc, uc)
    #                                for uc in x] * y)) * accu)
    # ]


def buyback(u, v, accu, price, cost, purchase_quantity, **kwargs):
    """
    在销售商家购买数额确定的情况下,先择最优的回购价格
    :param u: 均值/
    :type u:
    :param v:
    :type v:
    :param accu:
    :type accu:
    :param price:
    :type price:
    :param cost:
    :type cost:
    :param purchase_quantity:
    :type purchase_quantity:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """


def main():
    u = 30
    v = 10
    accu = 0.1
    price = 20
    cost = 15
    
    supplier_price = 15
    supplier_cost = 8
    
    # 确定回购价格的方式
    # 保持利润比不变
    purchase_quantity, purchase_profit = newboys(u=u,
                                                 v=v,
                                                 accu=accu,
                                                 price=price,
                                                 cost=cost, price_buy_back=0,
                                                 title='销售商家利润分布图')
    
    # 从供应商+销售商家来看
    quantity, profit = newboys(u=u, v=v, price=price,
                               accu=accu,
                               cost=supplier_cost,
                               price_buy_back=0, title='供应商+销售商家利润分布图')
    assert purchase_quantity <= quantity
    
    # 计算供应商的利润
    supplier_quantity = purchase_quantity
    supplier_profit = supplier_quantity * (supplier_price - supplier_cost)
    print('销售商利润: ', round(purchase_profit, 1), '进货量: ', purchase_quantity)
    print('按此计划进货')
    print('供应商利润: ', supplier_profit)
    totalprofit = supplier_profit+purchase_profit
    print('总利润: ', round(totalprofit, 1))
    print('最优利润比为: ', round(supplier_profit/totalprofit, 4), round(purchase_profit/totalprofit, 4))
    print('按照最优利润比计算得回购价为: ', round(supplier_cost+(price-supplier_cost)*supplier_profit/totalprofit, 1))


if __name__ == '__main__':
    main()
