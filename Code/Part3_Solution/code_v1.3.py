# coding:utf-8

'''
# Author: chenhao
# Date: Aug.15.2018
# Description: site selecting (Optimum) using the simulated annealing algorithm 
'''

from random import random
import math
import sys
from time import time
import numpy as np
import pandas as pd


class SimAnneal(object):
    '''
    Simulated annealing algorithm 
    '''
    def __init__(self, target_text='min'):
        self.target_text = target_text

    def newVar(self, oldList, T):
        '''
        :old : list
        :return : list, new solutions based on old solutions
        :T   : current temperature
        '''
        newList = [i + (random()*2-1) for i in oldList]
        return newList

    def juge(self, func, new, old, T):
        '''
        matropolise conditions: to get the maximun or minmun
        :new : new solution data from self.newX
        :old : old solution data
        :T   : current temperature

        '''
        dE = func(new) - func(old) if self.target_text == 'max' else func(old) - func(new)
        if dE >= 0:
            x, ans = new, func(new)
        else:
            if math.exp(dE/T) > random():
                x, ans = new,func(new)
            else:
                x, ans = old, func(old)
        return [x, ans]

class OptSolution(object):
    '''
    find the optimal solution.

    '''
    def __init__(self, temperature0=100, temDelta=0.98,
                 temFinal=1e-8, Markov_chain=2000,
                 result=0, val_nd=[0]):
        # initial temperature
        self.temperature0 = temperature0
        # step factor for decreasing temperature
        self.temDelta = temDelta
        # the final temperature
        self.temFinal = temFinal
        # the Markov_chain length (inner loops numbers)
        self.Markov_chain = Markov_chain
        # the final result
        self.result = result
        # the initial coordidate values: 1D [0], 2D [0,0] ...
        self.val_nd = val_nd

    def mapRange(self, oneDrange):
        return (oneDrange[1]-oneDrange[0])*random() + oneDrange[0]


    def soulution(self, SA_newV, SA_juge, juge_text, ValueRange, func):
        '''
        calculate the extreme value: max or min value
        :SA_newV : function from class SimAnneal().newVar
        :SA_juge : function from class SimAnneal().juge_*
        :ValueRange : [], range of variables, 1D or 2D or 3D...
        :func : target function obtained from user

        '''
        Ti = self.temperature0
        ndim = len(ValueRange)
        f = max if juge_text=='max' else min
        loops = 0

        while Ti > self.temFinal:
            res_temp, val_temp = [], []
            preV = [[self.mapRange(ValueRange[j]) for i in range(self.Markov_chain)] for j in range(ndim)]
            newV = [SA_newV(preV[j], T=Ti) for j in range(ndim)]

            for i in range(self.Markov_chain):
                boolV = True
                for j in range(ndim):
                    boolV &= (ValueRange[j][0]<= newV[j][i] <= ValueRange[j][1])
                if boolV == True:
                    res_temp.append(SA_juge(new=[newV[k][i] for k in range(ndim)], func=func, old=[preV[k][i] for k in range(ndim)], T=Ti)[-1])
                    val_temp.append(SA_juge(new=[newV[k][i] for k in range(ndim)], func=func, old=[preV[k][i] for k in range(ndim)], T=Ti)[0])
                else:
                    continue
                loops += 1

            # get the index of extreme value
            idex = res_temp.index(f(res_temp))
            result_temp = f(self.result, f(res_temp))
            # update the cooordidate of current extrema value
            self.val_nd = self.val_nd if result_temp == self.result else val_temp[idex]
            # update the extreme value
            self.result = result_temp
            # update the current temperature
            Ti *= self.temDelta
            #print(self.val_nd, self.result)
        result = []
        result.append(self.val_nd)
        print(result)
        #print(loops)

        #print('The extreme value = %f' %self.result[-1])

# 导入聚类中心数据
path = 'Data/8.Simulated Annealing/'
# 导入k=200时的情况
centroid_ins = pd.read_csv(path + 'Centroid_40.csv')
now_ins = pd.read_csv(path + 'now_school.csv')
# 导入现有设施的数据

# 逐行读取数据
row_centroid_ins = centroid_ins.shape[0]
row_now_ins = now_ins.shape[0]


# distance_cache用于缓存距离数据
name = ['lat', 'lng', 'distance']
distance_cache = pd.DataFrame(columns=name)
distance_cache['lat'] = now_ins['lat']
distance_cache['lng'] = now_ins['lng']

for i in range(row_centroid_ins):
    for j in range(row_now_ins):
        # 取出距离当前需求点最近的三个现有点
        distance = np.sqrt(np.square(centroid_ins.iloc[i, 0] - now_ins.iloc[j, 0]) + np.square(centroid_ins.iloc[i, 1] - now_ins.iloc[j, 1]))
        distance_cache.iloc[j, 2] = distance
    # 找到当前需求点的所有距离数据后，对其进行升序排列
    distance_cache = distance_cache.sort_values(by=["distance"])
    # 利用模拟退火方法计算最优解
    def func2(w):
        x, y = w
        fxy = 0.1 * np.sqrt(np.square(x - distance_cache.iloc[0, 0]) + np.square(y - distance_cache.iloc[0, 1])) + 0.1 * np.sqrt(np.square(x - distance_cache.iloc[1, 0]) + np.square(y - distance_cache.iloc[1, 1])) + 0.1 * np.sqrt(np.square(x - distance_cache.iloc[2, 0]) + np.square(y - distance_cache.iloc[2, 1])) - 0.7 * np.sqrt(np.square(x - centroid_ins.iloc[i, 0]) + np.square(y - centroid_ins.iloc[i, 1]))
        return fxy


    targ = SimAnneal(target_text='max')
    init = -sys.maxsize  # for maximun case
    # init = sys.maxsize # for minimun case
    # lat_add, lat_dec表示维度可行范围， lng_add, lng_dec表示经度可行范围

    xyRange = [[31, 32], [120, 121]]
    t_start = time()

    calculate = OptSolution(Markov_chain=1000, result=init, val_nd=[0, 0])
    output = calculate.soulution(SA_newV=targ.newVar, SA_juge=targ.juge, juge_text='max', ValueRange=xyRange,
                                 func=func2)

    t_end = time()
    print('Running %.4f seconds' % (t_end - t_start))



'''
    # 计算当前聚类的中心到每个现有点之间的距离
    for j in range(row_now_ins):
        # distance1表示所求点到现有点之间的距离， distance2表示所求点到需求点之间的距离
        # distance1/2前面的系数表示影响因子
        
        def func2(w):
            x, y = w
            fxy = 0.3*np.sqrt(np.square(x - now_ins.iloc[j, 0]) + np.square(y - now_ins.iloc[j, 1])) - 0.7*np.sqrt(np.square(x - centroid_ins.iloc[i, 0]) + np.square(y - centroid_ins.iloc[i, 1]))
            return fxy


        targ = SimAnneal(target_text='max')
        init = -sys.maxsize  # for maximun case
        # init = sys.maxsize # for minimun case
        # lat_add, lat_dec表示维度可行范围， lng_add, lng_dec表示经度可行范围

        xyRange = [[31, 32], [120, 121]]
        t_start = time()

        calculate = OptSolution(Markov_chain=1000, result=init, val_nd=[0, 0])
        output = calculate.soulution(SA_newV=targ.newVar, SA_juge=targ.juge, juge_text='max', ValueRange=xyRange, func=func2)

        t_end = time()
        print('Running %.4f seconds' % (t_end - t_start))
'''

