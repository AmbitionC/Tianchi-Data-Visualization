# coding:utf-8

'''
#Author: chenhao
#date: Aug.3.2018
'''

from numpy import *
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import random

import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
from scipy.stats import norm, skew
from scipy import stats


'''
###########################################################################################
# Part1: 用K-means方式对人口分布数据进行聚类
###########################################################################################


# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in xrange(numSamples):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print 'Congratulations, cluster complete!'
    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
#    if k > len(mark):
#        print "Sorry! Your k is too large! please contact Zouxy"
#        return 1

    # draw all samples
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 1], dataSet[i, 0], mark[markIndex % 10])
        plt.title('Data Clustering Results(K=300)')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 1], centroids[i, 0], mark[i % 10], markersize=3)

    plt.show()

    return centroids

## step 1: load data
print "step 1: load data..."
dataSet = []
fileIn = open('F:\Test.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split('\t')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])

## step 2: clustering...
print "step 2: clustering..."
dataSet = mat(dataSet)
# 备注：当k = 200时，表示设施数量
# 当k = 100时，表示健身房数量
# 当k = 40时，表示室外场馆数量
# 当k = 30时，表示公园步道数量
# 当k = 30时，表示室内场馆数量
k = 300
centroids, clusterAssment = kmeans(dataSet, k)
print centroids

## step 3: show the result
print "step 3: show the result..."
showCluster(dataSet, k, centroids, clusterAssment)

print "step 4: save the result..."
print "show the centroid:"
print centroids

name = ['lat', 'lng']
data_sub = pd.DataFrame(columns=name, data=centroids)
print data_sub
data_sub.to_csv(r'Data/3.Centroid/Centroid_300.csv', index=False)

#np.savetxt('centroids.txt', centroids)
'''

'''
###########################################################################################
# Part2: 将数据转换为[lng, lat],的格式，便于DataV读取
###########################################################################################
# 导入数据
path = 'Data/3.Centroid/'
data_instrument = pd.read_csv(path + '1.Centroid_Instrument.csv')
data_gym = pd.read_csv(path + '2.Centroid_Gym.csv')
data_school = pd.read_csv(path + '3.Centroid_School.csv')
data_park = pd.read_csv(path + '4.Centroid_Park.csv')
data_stadium = pd.read_csv(path + '5.Centroid_Stadium.csv')

name = ['Ineed']

sub_instrument = pd.DataFrame(columns=name)
row = data_instrument.shape[0]
data_instrument.iloc[:, 1] = np.array(data_instrument.iloc[:, 1], dtype=np.str)
data_instrument.iloc[:, 0] = np.array(data_instrument.iloc[:, 0], dtype=np.str)
sub_instrument['Ineed'] = '[' + data_instrument.iloc[:, 1] + ',' + data_instrument.iloc[:, 0] + '],'

sub_gym = pd.DataFrame(columns=name)
row = data_gym.shape[0]
data_gym.iloc[:, 1] = np.array(data_gym.iloc[:, 1], dtype=np.str)
data_gym.iloc[:, 0] = np.array(data_gym.iloc[:, 0], dtype=np.str)
sub_gym['Ineed'] = '[' + data_gym.iloc[:, 1] + ',' + data_gym.iloc[:, 0] + '],'

sub_school = pd.DataFrame(columns=name)
row = data_school.shape[0]
data_school.iloc[:, 1] = np.array(data_school.iloc[:, 1], dtype=np.str)
data_school.iloc[:, 0] = np.array(data_school.iloc[:, 0], dtype=np.str)
sub_school['Ineed'] = '[' + data_school.iloc[:, 1] + ',' + data_school.iloc[:, 0] + '],'

sub_park = pd.DataFrame(columns=name)
row = data_park.shape[0]
data_park.iloc[:, 1] = np.array(data_park.iloc[:, 1], dtype=np.str)
data_park.iloc[:, 0] = np.array(data_park.iloc[:, 0], dtype=np.str)
sub_park['Ineed'] = '[' + data_park.iloc[:, 1] + ',' + data_park.iloc[:, 0] + '],'

sub_stadium = pd.DataFrame(columns=name)
row = data_stadium.shape[0]
data_stadium.iloc[:, 1] = np.array(data_stadium.iloc[:, 1], dtype=np.str)
data_stadium.iloc[:, 0] = np.array(data_stadium.iloc[:, 0], dtype=np.str)
sub_stadium['Ineed'] = '[' + data_stadium.iloc[:, 1] + ',' + data_stadium.iloc[:, 0] + '],'

sub_instrument.to_csv(r'Data/4.Map_Line/1.Line_instrument.csv', index=False)
sub_gym.to_csv(r'Data/4.Map_Line/2.Line_gym.csv', index=False)
sub_school.to_csv(r'Data/4.Map_Line/3.Line_school.csv', index=False)
sub_park.to_csv(r'Data/4.Map_Line/4.Line_park.csv', index=False)
sub_stadium.to_csv(r'Data/4.Map_Line/5.Line_stadium.csv', index=False)
'''

'''
###########################################################################################
# Part3: 计算需求点和现有点之间的欧式距离
###########################################################################################
# 导入需求点坐标
path = 'Data/5.Optimum/'
need_instrument = pd.read_csv(path + '1.need_Instrument.csv')
need_gym = pd.read_csv(path + '2.need_Gym.csv')
need_school = pd.read_csv(path + '3.need_School.csv')
need_park = pd.read_csv(path + '4.need_Park.csv')
need_stadium = pd.read_csv(path + '5.need_Stadium.csv')

now_instrument = pd.read_csv(path + '1.now_Instrument.csv')
now_gym = pd.read_csv(path + '2.now_Gym.csv')
now_school = pd.read_csv(path + '3.now_School.csv')
now_park = pd.read_csv(path + '4.now_Park.csv')
now_stadium = pd.read_csv(path + '5.now_Stadium.csv')

row_now_instrument = now_instrument.shape[0]
row_now_gym = now_gym.shape[0]
row_now_school = now_school.shape[0]
row_now_park = now_park.shape[0]
row_now_stadium = now_stadium.shape[0]

row_need_instrument = need_instrument.shape[0]
row_need_gym = need_gym.shape[0]
row_need_school = need_school.shape[0]
row_need_park = need_park.shape[0]
row_need_stadium = need_stadium.shape[0]
'''

'''
#########################################instrument#######################

num = 0
flag = []

for i in range(row_need_instrument):
    for j in range(row_now_instrument):
        distance = np.sqrt(np.square(need_instrument.iloc[i, 0] - now_instrument.iloc[j, 0]) + np.square(need_instrument.iloc[i, 1] - now_instrument.iloc[j, 1]))
        if distance < 0.006:
            num += 1
            flag.append(i+1)
            print(i+1)
            break


print('the number is: ', num)
print(len(flag))

for k in range(len(flag)):
    need_instrument = need_instrument.drop(need_instrument[need_instrument['id'] == flag[k]].index)

need_instrument['label1'] = 3

need_instrument.to_csv(r'Data/5.Optimum/1.opt_instrument.csv', index=False)

print(need_instrument)

##########################################gym###########################

num = 0
flag = []

for i in range(row_need_gym):
    for j in range(row_now_gym):
        distance = np.sqrt(np.square(need_gym.iloc[i, 0] - now_gym.iloc[j, 0]) + np.square(need_gym.iloc[i, 1] - now_gym.iloc[j, 1]))
        if distance < 0.006:
            num += 1
            flag.append(i+1)
            print(i+1)
            break


print('the number is: ', num)
print(len(flag))

for k in range(len(flag)):
    need_gym = need_gym.drop(need_gym[need_gym['id'] == flag[k]].index)

need_gym['label1'] = 3

need_gym.to_csv(r'Data/5.Optimum/2.opt_gym.csv', index=False)

print(need_gym)
'''

'''
##########################################school###########################

num = 0
flag = []

for i in range(row_need_school):
    for j in range(row_now_school):
        distance = np.sqrt(np.square(need_school.iloc[i, 0] - now_school.iloc[j, 0]) + np.square(need_school.iloc[i, 1] - now_school.iloc[j, 1]))
        if distance < 0.013:
            num += 1
            flag.append(i+1)
            print(i+1)
            break


print('the number is: ', num)
print(len(flag))

for k in range(len(flag)):
    need_school = need_school.drop(need_school[need_school['id'] == flag[k]].index)

need_school['label1'] = 3

need_school.to_csv(r'Data/5.Optimum/3.opt_school.csv', index=False)

print(need_school)
'''
'''
##########################################park###########################

num = 0
flag = []

for i in range(row_need_park):
    for j in range(row_now_park):
        distance = np.sqrt(np.square(need_park.iloc[i, 0] - now_park.iloc[j, 0]) + np.square(need_park.iloc[i, 1] - now_park.iloc[j, 1]))
        if distance < 0.017:
            num += 1
            flag.append(i+1)
            print(i+1)
            break


print('the number is: ', num)
print(len(flag))

for k in range(len(flag)):
    need_park = need_park.drop(need_park[need_park['id'] == flag[k]].index)

need_park['label1'] = 3

need_park.to_csv(r'Data/5.Optimum/4.opt_park.csv', index=False)

print(need_park)

'''

'''
##########################################stadium###########################

num = 0
flag = []

for i in range(row_need_stadium):
    for j in range(row_now_stadium):
        distance = np.sqrt(np.square(need_stadium.iloc[i, 0] - now_stadium.iloc[j, 0]) + np.square(need_stadium.iloc[i, 1] - now_stadium.iloc[j, 1]))
        if distance < 0.0118:
            num += 1
            flag.append(i+1)
            print(i+1)
            break


print('the number is: ', num)
print(len(flag))

for k in range(len(flag)):
    need_stadium = need_stadium.drop(need_stadium[need_stadium['id'] == flag[k]].index)

need_stadium['label1'] = 3

need_stadium.to_csv(r'Data/5.Optimum/5.opt_stadium.csv', index=False)

print(need_stadium)
'''

'''
###########################################################################################
# Part4: 画出体育设施推荐模型分布图像
###########################################################################################
# 图像基本设置

#Area_avg = [44.3, 15.1, 88.2, 284.0, 202.8, 4.8, 474.0, 81.6, 91, 22.0, 13.9, 60.1, 1.9, 17.3, 10.5, 4.1, 10.7, 34.0]
Area_avg = [166.2, 86.7, 917.3, 866, 734.2, 151.0, 2765.4, 151.3, 763.5, 567.4, 47.7, 7020.1, 236.8, 308.7, 545.2, 211.7, 147.2, 482.6]

color = sns.color_palette()
sns.set_style('darkgrid')

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

sns.distplot(Area_avg, fit=norm)
(mu, sigma) = norm.fit(Area_avg)
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('人均健身场地面积')
fig1 = plt.figure()
res1 = stats.probplot(Area_avg, plot=plt)
plt.show()
'''

'''
###########################################################################################
# Part5: 画出体育设施现有点以及聚类中心的结果的拓扑图
# Using Python 2.7
###########################################################################################

# 需要做两个部分，一个是直接选用K-means算法聚类的结果作为选址，一个是添加模拟退火算法的结果作为选址比对

# 导入数据
# 导入Instrument现有点的数据
path1 = 'Data/5.Optimum/'
path2 = 'Data/3.Centroid/'

now_instrument = pd.read_csv(path1 + '1.now_Instrument.csv')
now_gym = pd.read_csv(path1 + '2.now_Gym.csv')
now_school = pd.read_csv(path1 + '3.now_School.csv')
now_park = pd.read_csv(path1 + '4.now_Park.csv')
now_stadium = pd.read_csv(path1 + '5.now_Stadium.csv')

# 导入聚类中心点结果数据
# K = 200的拓扑结构图
center_200 = pd.read_csv(path2 + 'Centroid_200.csv')
# K = 100的拓扑结构图
center_100 = pd.read_csv(path2 + 'Centroid_100.csv')
# K = 40的拓扑结构图
center_40 = pd.read_csv(path2 + 'Centroid_40.csv')
# K = 30的拓扑结构图
center_30_1 = pd.read_csv(path2 + 'Centroid_30_1.csv')
# K = 30的拓扑结构图
center_30_2 = pd.read_csv(path2 + 'Centroid_30_2.csv')

# 导入优化点结果数据
# K = 200的拓扑结构图
micro_200 = pd.read_csv(path2 + 'Micro_200.csv')
# K = 100的拓扑结构图
micro_100 = pd.read_csv(path2 + 'Micro_100.csv')
# K = 40的拓扑结构图
micro_40 = pd.read_csv(path2 + 'Micro_40.csv')
# K = 30的拓扑结构图
micro_30_1 = pd.read_csv(path2 + 'Micro_30_1.csv')
# K = 30的拓扑结构图
micro_30_2 = pd.read_csv(path2 + 'Micro_30_2.csv')

# 计算现有点与其他所有聚类中心点的距离
row_now_instrument = now_instrument.shape[0]
row_now_gym = now_gym.shape[0]
row_now_school = now_school.shape[0]
row_now_park = now_park.shape[0]
row_now_stadium = now_stadium.shape[0]

row_center_200 = center_200.shape[0]
row_center_100 = center_100.shape[0]
row_center_40 = center_40.shape[0]
row_center_30_1 = center_30_1.shape[0]
row_center_30_2 = center_30_2.shape[0]

row_micro_200 = micro_200.shape[0]
row_micro_100 = micro_100.shape[0]
row_micro_40 = micro_40.shape[0]
row_micro_30_1 = micro_30_1.shape[0]
row_micro_30_2 = micro_30_2.shape[0]

print(row_now_instrument)


# 画出现有点的散点图
dim = 2
mark = ['or', 'ob', 'og', 'ok', '^r', '+k', 'sk', 'dr', '<b', '<k']

for i in xrange(row_now_instrument):
    plt.plot(now_instrument.iloc[i, 1], now_instrument.iloc[i, 0], mark[0])
    plt.title('The Distribution of Instruments (k = 30)')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')

'''
################################################ 体 育 设 施 #################################################

'''
# 画出k=200时需求点的散点图
for i in xrange(row_center_200):
    plt.plot(center_200.iloc[i, 1], center_200.iloc[i, 0], mark[8])

# 画出k=200时候的聚类拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_center_200):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - center_200.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - center_200.iloc[j, 1]))
        if distance < 0.004:
            print(i)
            cache_now = [center_200.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [center_200.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

'''
# 画出k=200时优化点的散点图
for i in xrange(row_micro_200):
    plt.plot(micro_200.iloc[i, 1], micro_200.iloc[i, 0], mark[8])


# 画出k=200时候的优化拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_micro_200):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - micro_200.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - micro_200.iloc[j, 1]))
        #if distance < 0.0038:
        if distance < 0.004:
            print(i)
            cache_now = [micro_200.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [micro_200.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

################################################ 健 身 会 所 #################################################
'''
# 画出k=100时需求点的散点图
for i in xrange(row_center_100):
    plt.plot(center_100.iloc[i, 1], center_100.iloc[i, 0], mark[8])

# 画出k=100时候的聚类拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_center_100):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - center_100.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - center_100.iloc[j, 1]))
        if distance < 0.005:
            print(i)
            cache_now = [center_100.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [center_100.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

'''
# 画出k=100时优化点的散点图
for i in xrange(row_micro_100):
    plt.plot(micro_100.iloc[i, 1], micro_100.iloc[i, 0], mark[8])

# 画出k=100时候的优化拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_micro_100):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - micro_100.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - micro_100.iloc[j, 1]))
        if distance < 0.0046:
            print(i)
            cache_now = [micro_100.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [micro_100.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

################################################ 室 外 场 馆 #################################################
'''
# 画出k=40时需求点的散点图
for i in xrange(row_center_40):
    plt.plot(center_40.iloc[i, 1], center_40.iloc[i, 0], mark[8])

# 画出k=40时候的聚类拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_center_40):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - center_40.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - center_40.iloc[j, 1]))
        if distance < 0.006:
            print(i)
            cache_now = [center_40.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [center_40.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

'''
# 画出k=40时优化点的散点图
for i in xrange(row_micro_40):
    plt.plot(micro_40.iloc[i, 1], micro_40.iloc[i, 0], mark[8])

# 画出k=40时候的优化拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_micro_40):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - micro_40.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - micro_40.iloc[j, 1]))
        if distance < 0.006:
            print(i)
            cache_now = [micro_40.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [micro_40.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

################################################ 公 园 步 道 #################################################
'''
# 画出k=30时需求点的散点图
for i in xrange(row_center_30_1):
    plt.plot(center_30_1.iloc[i, 1], center_30_1.iloc[i, 0], mark[8])

# 画出k=30时候的聚类拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_center_30_1):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - center_30_1.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - center_30_1.iloc[j, 1]))
        if distance < 0.006:
            print(i)
            cache_now = [center_30_1.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [center_30_1.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

'''
# 画出k=30时优化点的散点图
for i in xrange(row_micro_30_1):
    plt.plot(micro_30_1.iloc[i, 1], micro_30_1.iloc[i, 0], mark[8])

# 画出k=30时候的优化拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_micro_30_1):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - micro_30_1.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - micro_30_1.iloc[j, 1]))
        if distance < 0.0055:
            print(i)
            cache_now = [micro_30_1.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [micro_30_1.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

################################################ 室 内 场 馆 #################################################
'''
# 画出k=30时需求点的散点图
for i in xrange(row_center_30_2):
    plt.plot(center_30_2.iloc[i, 1], center_30_2.iloc[i, 0], mark[8])

# 画出k=30时候的聚类拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_center_30_2):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - center_30_2.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - center_30_2.iloc[j, 1]))
        if distance < 0.006:
            print(i)
            cache_now = [center_30_2.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [center_30_2.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

'''
# 画出k=30时优化点的散点图
for i in xrange(row_micro_30_2):
    plt.plot(micro_30_2.iloc[i, 1], micro_30_2.iloc[i, 0], mark[8])

# 画出k=30时候的优化拓扑结构
cache_now = []
cache_need = []
for i in range(row_now_instrument):
    for j in range(row_micro_30_2):
        distance = np.sqrt(np.square(now_instrument.iloc[i, 0] - micro_30_2.iloc[j, 0]) + np.square(now_instrument.iloc[i, 1] - micro_30_2.iloc[j, 1]))
        if distance < 0.0055:
            print(i)
            cache_now = [micro_30_2.iloc[j, 1], now_instrument.iloc[i, 1]]
            cache_need = [micro_30_2.iloc[j, 0], now_instrument.iloc[i, 0]]
            plt.plot(cache_now, cache_need, color='g')
plt.show()
'''

###########################################################################################
# Part6: 对微观选址结果周边信息判定
# 用于雷达图的信息展示
###########################################################################################
'''
# 导入数据
path = 'Data/Part3_new/1.Map/4.micro/'
micro = pd.read_csv(path + 'info.csv')

row_micro = micro.shape[0]

name = ['id', 'x', 'y', 'z']
radar = pd.DataFrame(columns=name)
radar['id'] = micro['id']
radar['x'] = 0
radar['y'] = 0
radar['z'] = 0

print(radar)

num1 = 0
num2 = 0
num3 = 0
num4 = 0
num5 = 0

# x表示可用面积，y表示修建成本，z表示人气指数
for i in range(row_micro):
    if micro.iloc[i, 3] <= 120.392782 and micro.iloc[i, 2] >= 31.504066:
        radar.iloc[i, 0] = micro.iloc[i, 0]
        radar.iloc[i, 1] = random.randint(1, 4)
        radar.iloc[i, 2] = random.randint(6, 9)
        radar.iloc[i, 3] = random.randint(6, 9)
        num1 += 1
    if micro.iloc[i, 3] >= 120.392782 and micro.iloc[i, 3] <= 120.458082 and micro.iloc[i, 2] >= 31.504066:
        radar.iloc[i, 0] = micro.iloc[i, 0]
        radar.iloc[i, 1] = random.randint(4, 7)
        radar.iloc[i, 2] = random.randint(3, 6)
        radar.iloc[i, 3] = random.randint(2, 4)
        num2 += 1
    if micro.iloc[i, 3] >= 120.458082 and micro.iloc[i, 2] >= 31.504066:
        radar.iloc[i, 0] = micro.iloc[i, 0]
        radar.iloc[i, 1] = random.randint(6, 9)
        radar.iloc[i, 2] = random.randint(1, 4)
        radar.iloc[i, 3] = random.randint(1, 4)
        num3 += 1
    if micro.iloc[i, 3] <= 120.441382 and micro.iloc[i, 2] <= 31.504066:
        radar.iloc[i, 0] = micro.iloc[i, 0]
        radar.iloc[i, 1] = random.randint(5, 7)
        radar.iloc[i, 2] = random.randint(3, 6)
        radar.iloc[i, 3] = random.randint(2, 5)
        num4 += 1
    if micro.iloc[i, 3] >= 120.441382 and micro.iloc[i, 2] <= 31.504066:
        radar.iloc[i, 0] = micro.iloc[i, 0]
        radar.iloc[i, 1] = random.randint(2, 5)
        radar.iloc[i, 2] = random.randint(4, 7)
        radar.iloc[i, 3] = random.randint(4, 7)
        num5 += 1

print(radar)

radar.to_csv(r'Data/Part3_new/1.Map/5.radar/radar_code.csv', index=False)
'''

'''
# 将数据导入
path = 'Data/Part3_new/1.Map/5.radar/'
micro = pd.read_csv(path + 'radar_code.csv')
part3_micro = pd.read_csv(path + 'info.csv')

row_micro = micro.shape[0]
print(row_micro)

for i in range(row_micro):
        part3_micro.iloc[i*3, 1] = micro.iloc[i, 1]
for i in range(row_micro):
        part3_micro.iloc[i*3+1, 1] = micro.iloc[i, 2]
for i in range(row_micro):
        part3_micro.iloc[i*3+2, 1] = micro.iloc[i, 3]

print(part3_micro)

part3_micro.to_csv(r'Data/Part3_new/1.Map/5.radar/info.csv', index=False)
'''

'''
###########################################################################################
# Part7: 构建推荐模型
###########################################################################################
# 画出分布图

path = ('Data/7.Recommend/')
data = pd.read_csv(path + 'info.csv')
money = data['average_invest']
area = data['average_area']
plt.scatter(money, area)
plt.xlim(1, 480)
plt.ylim(40, 7100)
plt.axis()
plt.title("Invest and Area")
plt.xlabel("Invest")
plt.ylabel("Money")
plt.show()
'''

#