# coding:utf-8

'''
# Author: chenhao
# Date: Aug.12.2018
# Description: Selecting the number of the instruments using Genetic Algorithm
'''



from math import sin as sin
from math import cos as cos
from math import exp as exp
from random import random as rand
from random import shuffle as shuffle
from copy import copy
import numpy as np




rg = [[0, 50], [0, 50], [0, 50], [0, 50], [0, 50]]  # 搜索范围
ST = 50  # 约束条件
STC = 45
N = 150  # 初始种群规模
E = 0.03  # 适应度偏移修正
e = 0.01  # 变异基础值
T = 4  # 淘汰比率/每个个体平均产生配子数
n = 5  # 输入变量数/染色体数/修改这里以实现不同的模型
# g = 1000     # 输入空间分割数/决定单个性状的基因数（线性变异用）
g1 = 20  # 输入空间分割数/决定单个性状的基因数
e1 = 0.30  # 基础变异比率
times = 80  # 迭代次数/引入平方项后收敛速度很快
psw = 0.5  # 交换发生概率
T = T / 2  # 淘汰比率修正

# 输入相应的权重
W1 = 0.7
W2 = 0.15
W3 = 0.15
# 输入相应的需求强度、修建成本、区域内已有设施数量

# alpha1i表示需求强度，系数与住宅区数量成正比
alpha11 = 33
alpha12 = 14
alpha13 = 4
alpha14 = 17
alpha15 = 11

# alpha2i表示修建成本，系数与绿化面积成反比
alpha21 = 62
alpha22 = 42
alpha23 = 10
alpha24 = 49
alpha25 = 25

# alpha3i表示已有设施影响程度，系数与区域内已有设施数量统计成正比
alpha31 = 52
alpha32 = 12
alpha33 = 5
alpha34 = 30
alpha35 = 25

# 目标函数/修改这里以实现不同的模型
def y(x):
    #return -W1 * (alpha11 * x[0] + alpha12 * x[1] + alpha13 * x[2] + alpha14 * x[3] + alpha15 * x[4]) + W2 * (
    #alpha21 * x[0] + alpha22 * x[1] + alpha23 * x[2] + alpha24 * x[3] + alpha25 * x[4]) + W3 * (
    #alpha31 * x[0] + alpha32 * x[1] + alpha33 * x[2] + alpha34 * x[3] + alpha35 * x[4])

    return -W1*(alpha11*exp(-0.5*x[0]) + alpha12*exp(-0.5*x[1]) + alpha13*exp(-0.5*x[2]) + alpha14*exp(-0.5*x[3]) + alpha15*exp(-0.5*x[4])) + W2*(alpha21*exp(-0.5*x[0]) + alpha22*exp(-0.5*x[1]) + alpha23*exp(-0.5*x[2]) + alpha24*exp(-0.5*x[3]) + alpha25*exp(-0.5*x[4])) + W3*(alpha31*exp(-0.5*x[0]) + alpha32*exp(-0.5*x[1]) + alpha33*exp(-0.5*x[2]) + alpha34*exp(-0.5*x[3]) + alpha35*exp(-0.5*x[4]))



    #return -0.3*80*(exp(-0.5*x[0])) + 0.3*30*(exp(-0.5*x[0])) - 0.3*30*(1/exp(-0.5*x[0])) - 0.3*60*(exp(-0.5*x[1])) + 0.3*40*(exp(-0.5*x[1])) - 0.3*70*(1/exp(-0.5*x[1])) - 0.3*20*(exp(-0.5*x[2])) + 0.3*10*(exp(-0.5*x[2])) - 0.3*20*(1/exp(-0.5*x[2]))
    #return sin(x[1]) * exp(1 - cos(x[0])) ** 2 + cos(x[0]) * exp(1 - sin(x[1])) ** 2 + (x[0] - x[1]) ** 2



# 约束函数/修改这里以实现不同的模型
def st(x):
    #if x[0] + x[1] + x[2] <= ST and x[0] >= 0 and x[1] >= 0 and x[2] >=0 :
    if x[0] + x[1] + x[2] + x[3] + x[4] <= ST and x[0] >= 0 and x[1] >= 0 and x[2] >= 0 and x[3] >= 0 and x[4] >= 0 and x[0] + x[1] + x[2] + x[3] + x[4] >= STC:
        return True
    return False


# 生成初始种群
def initiate():
    M = [[0.0 for j in range(n)] for i in range(N)]
    num = 0
    while num < N:
        m = [rand() * (j[1] - j[0]) + j[0] for j in rg]
        if st(m):
            M[num] = m
            num = num + 1
    return M


# 评估适应度
def fitting(Y):
    y_max = max(Y)
    f = [(y_max - yi) ** 2 + E for yi in Y]
    return [j / sum(f) for j in f]


# 产生配合子，注：使用乘法修正概率密度函数远快于遍历一个向量
def reproduction(F, M):
    P = []
    for i in range(N):
        for j in range(F[i]):
            P.append(M[i])
    return P


# 变异处理,F为配合子数目向量，生成一个变异矩阵（线性变异，已经停用，可用于对照组）
def mutate(F, Y):
    y_min = min(Y)
    pe = [(y_min - yi) ** 2 + e for yi in Y]
    pe = [j / sum(pe) for j in pe]  # 对优等个体进行保护，利用二次项减少变异概率
    mutt = []
    for i in range(N):
        for j in range(F[i]):
            mt = [poisson(pe[i] * g) * (round(rand() - 0.001) * 2 - 1) * (rg[k][1] - rg[k][0]) / g for k in range(n)]
            mutt.append(mt)
    return mutt


# 变异处理,F为配合子数目向量，生成一个变异矩阵（2-指数变异）
def mutate1(F, Y):
    y_min = min(Y)
    pe = [(y_min - yi) ** 2 + e for yi in Y]
    pe = [j * N * e1 / (sum(pe)) for j in pe]  # 对优等个体进行保护，利用二次项减少变异概率
    mutt = []
    for i in range(N):
        for j in range(F[i]):
            mt = [(2 ** poisson(pe[i] * g1)) * (round(rand() - 0.001) * 2 - 1) * (rg[k][1] - rg[k][0]) / (2 ** g1) for k
                  in range(n)]
            mutt.append(mt)
    return mutt


# 生成泊松分布随机数，方法：Knuth方法，拟合二项分布对变异数目进行估计，期望lambda = np
def poisson(l):
    L = float(exp(-l))
    k = 0.0
    p = 1.0
    while 1:
        k = k + 1
        p = p * rand()
        if p < L:
            break
    return k - 1


# 交叉互换
def switch(P, Pi):
    sw = [int(rand() <= psw) for t in range(n)]
    for i in range(len(P)):
        for j in range(n):
            temp = P[i][j]
            P[i][j] = Pi[i][j] * sw[j] + P[i][j] * (1 - sw[j])
            Pi[i][j] = temp * sw[j] + Pi[i][j] * (1 - sw[j])
    return P + Pi


# GA主函数
def genetic_algorithm():
    M = initiate()  # 随机生成初始种群
    elite = M[0]  # 精英初始化
    my = y(elite)  # 精英适应度初始化

    for t in range(times):
        Y = [y(m) for m in M]  # 表现型/输出
        f = fitting(Y)  # 适应度

        if min(Y) < my:  # 精英竞争
            my = min(Y)
            idx = Y.index(my)
            elite = M[idx]  # 精英个体保护

        F = [int(round(j * T * N)) for j in f]  # 个体生成配合子个数/繁殖次数

        P = reproduction(F, M)  # 生成繁殖序列
        mutt = mutate1(F, Y)  # 变异矩阵

        P = map(lambda (a, b): map(lambda (l1, l2): l1 + l2, zip(a, b)), zip(P, mutt))  # 执行变异

        i = 0      # 个体可行性检查/可行域检查
        while i < len(P):
            if ~st(P[i]):
                P.pop(i)
            i = i + 1

        Pi = copy(P)
        shuffle(P)  # 打乱组合1
        shuffle(Pi)  # 打乱组合2

        Mi = switch(P, Pi) + initiate()  # 进入种群的外来个体
        Mi.append(elite)  # 精英保留于种群

        Yi = [y(m) for m in Mi]
        fi = fitting(Yi)
        M = []
        c = 0
        while c < N:     # 从子代优选个体成为新一代父代
            ym = max(fi)
            idx = fi.index(ym)
            fi[idx] = 0
            if st(Mi[idx]):
                M.append(Mi[idx])
                c = c + 1

    Y = [y(m) for m in M]  # 表现型/输出
    f = fitting(Y)  # 适应度
    ym = max(f)
    idx = f.index(ym)
    return M[idx], Y[idx]


M, Y = genetic_algorithm()
print M, Y

'''
    return -W1 * (alpha11 * np.square(0.2965 * x[0]) + alpha12 * np.square(0.2965 * x[1]) + alpha13 * np.square(0.2965 * x[2]) + alpha14 * np.square(
        0.2965 * x[3]) + alpha15 * np.square(0.2965 * x[4])) + W2 * (
    alpha21 * exp(0.09 * x[0]) + alpha22 * exp(0.09 * x[1]) + alpha23 * exp(0.09 * x[2]) + alpha24 * exp(
        0.09 * x[3]) + alpha25 * exp(0.09 * x[4])) + W3 * (
    alpha31 * (1.2 * x[0]) + alpha32 * (1.2 * x[1]) + alpha33 * (1.2 * x[2]) + alpha34 * (
        1.2 * x[3]) + alpha35 * (1.2 * x[4]))
    '''


'''
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(211)
# original
# f = lambda x: -0.33*33*(exp(-0.5*x)) + 0.33*62*(exp(-0.5*x)) - 0.3*52*(1/(exp(-0.5*x)))
f = lambda x: -0.33*33*(np.square(0.28*x)) + 0.33*62*(exp(0.09*x)) + 0.33*52*(exp(0.09*x))
#f = lambda x: np.square(0.2*x)
x = np.linspace(0, 50, 1000)
y = [f(i) for i in x]
ax.plot(x, y)
plt.show()
'''