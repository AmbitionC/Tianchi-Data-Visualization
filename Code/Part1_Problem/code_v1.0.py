# coding:utf-8

'''
#Author: chenhao
#date: July.20.2018
'''

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import math
from pylab import mpl
import seaborn as sns
from scipy.stats import norm, skew
from scipy import stats

'''
###########################################################################################
# Part1: 将全国的国民体质监测达标率数据进行转换，其中包括全国各省，江苏省各市，无锡市各区的情况
###########################################################################################
# 导入数据
data_path = 'Data/'
data_origin = pd.read_csv(data_path + 'Health_China.csv', encoding='gb2312')

# 导出数据
data_sub = pd.DataFrame()
data_sub['area_id'] = data_origin['dis_code']
data_sub['value'] = data_origin['Total']

# 存储数据
data_sub.to_csv(r'Data/Health_label.csv', index=False)

# 实验结论：产生的数据可视化结果不明显，在全国地图上颜色较为统一。
'''

'''
###########################################################################################
# Part2: 在Part1的基础上添加差分，同时保留原始数据，将省份信息和合格率添加到info字段
###########################################################################################
# 导入数据
data_path = 'Data/'
data_origin = pd.read_csv(data_path + 'Health_China.csv', encoding='gb2312')

# 导出数据
data_sub = pd.DataFrame()
data_sub['area_id'] = data_origin['dis_code']
data_sub['cache'] = data_origin['Total']
data_sub['value'] = 0

# 对体质监测达标率数据逐行修改
rows = data_sub['cache'].shape[0]
for i in range(rows):
    data_sub.iloc[i, 2] = (100 - data_sub.iloc[i, 1]) * 5
data_sub = data_sub.drop(['cache'], axis=1)

# 存储数据
data_sub.to_csv(r'Data/Health_sub.csv', index=False)

# 实验结论：下钻热力图在使用的过程中，省级单位显示时，会出现都是一种颜色的情况
# 解决方案：添加一层热力图，用以显示省级单位，保留下钻热力图及其原始数据
# 解决方案2：将数据的省、市、区、标签分别用不同的图层进行添加
'''


'''
###########################################################################################
# Part3: 采用ARIMA和ARMA进行时序预测，输入为1995年至2013年设备数量以及健身路径长度，预测2014年至2020年七年间的数据
###########################################################################################
# 导入数据
path = 'Data/4.ARIMA/'
data_origin = pd.read_csv(path + 'History.csv', encoding='gb2312')
data_pre = pd.read_csv(path + 'History_Pre.csv', encoding='gb2312')

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['FangSong']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 画出设备数量和运动路径的长度
plt.title('设备数时序图')
plt.ylabel('数量')
plt.xlabel('年份')
x = range(len(data_origin['Year']))
y1 = data_origin['Instrument']
y2 = data_origin['Path']
y3 = data_origin['Ins_total']
y4 = data_origin['Path_total']
y5 = data_pre['Instrument']
x1 = data_pre['Year']

#plt.plot(x, y1, label='Instrument')
plt.plot(x1, y5, label='Prediction')
#plt.plot(x, y2, label='Path')
plt.xticks(x1, data_pre['Year'], rotation=45)
plt.show()

# 做出设备数量的自相关图
plot_acf(y1)
plt.show()

# 平稳性检测
print(u'原始序列的ADF检验结果为：', ADF(y1))
# 返回值依次为adf、pvalue、nobs、critical values、icbest、regresult、resstore
# adf:-0.0
# pvalue: 0.95853208606005602
# nobs: 8
# critical values: 10
# icbest: {'1%': -4.3315729999999997, '5%': -3.2329500000000002, '10%': -2.7486999999999999}
# resstore: -414.96637673426136

# 对数据进行差分
D_data = y1.diff(1).dropna()
D_data.columns = [u'数目差分']
D_data.plot()
plt.show()

# 做出设备数量差分的自相关图
plot_acf(D_data)
plt.show()

# 做出设备数量差分的偏自相关图
plot_pacf(D_data)
plt.show()

# 设备数量差分平稳性检测
print(u'差分序列的ADF检验结果为：', ADF(D_data))
# 返回值依次为adf、pvalue、nobs、critical values、icbest、regresult、resstore
# adf:-3.8202312222795336
# pvalue: 0.0027069749201551079
# nobs: 7
# critical values: 10
# icbest: {'1%': -4.3315729999999997, '5%': -3.2329500000000002, '10%': -2.7486999999999999}
# resstore: -424.34713555026133

# 白噪声检验，返回统计量和p值
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))
# 统计量为4.52828986
# p值为0.03333892


# 定阶
# 一般阶数不超过length/10
D_data = y1.diff(1).dropna()
D_data.columns = [u'数目差分']
pmax = int(len(D_data)/10)
qmax = int(len(D_data)/10)

data_sub = data_origin['Instrument'].astype(float)
data_sub = pd.Series(data_sub)
data_sub.index = pd.Index(sm.tsa.datetools.dates_from_range('1995', '2013'))
print(data_sub)

# 定义bic矩阵
bic_matrix = []
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
        try:   #存在部分报错，所以用try来跳过报错
            tmp.append(ARIMA(data_sub, (p, 1, q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

# 从中可以找出最小值
bic_matrix = pd.DataFrame(bic_matrix)
print(bic_matrix)
print(bic_matrix.stack())

# 先用stack展平，然后用idxmin找出最小值位置
p, q = bic_matrix.stack().idxmin()
print(u'BIC最小的p值和q值为：%s、%s' %(p, q))

model = ARIMA(data_sub, (1, 1, 0)).fit(disp=0)
output = model.forecast(7)
print(output)
# 预测出来的结果为
# 6578.21705698,
# 6484.14868549,
# 7058.91598649,
# 7263.14857694,
# 7672.65725742,
# 7968.44306137,
# 8327.23129606.
'''


###########################################################################################
# Part4: 计算国民体检合格率和各区健身设施数目、人均健身设施数目、人均健身设施面积之间的关联性
###########################################################################################
# 导入数据
# health表示国民体检合格率，顺序依次为无锡市，江阴市，惠山区，锡山区，梁溪区，新吴区，滨湖区，宜兴市
health = [95.2, 92.3, 92.4, 86.4, 93, 84, 97.4, 94.3]
# Ins表示各个区的健身设施的数目
Ins = [14192, 4658, 1791, 1080, 1211, 1326, 1230, 2896]
# Ins_avg表示人均健身设施数目
Ins_avg = [21.89, 28.53, 25.49, 15.44, 12.73, 23.84, 17.70, 23.17]
# Area_avg表示人均健身设施面积
Area_avg = [2966.56, 3343.83, 2767.21, 1346.43, 2137.54, 3665.39, 5363.77, 2473.56]

# 计算平均数
def mean(x):
    return sum(x) / len(x)

# 计算每一项数据与均值的差
def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

# 辅助计算函数dot product, sum_of_squares
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    return dot(v, v)

# 计算方差
def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

# 标准差
def standard_deviation(x):
    return math.sqrt(variance(x))

# 计算协方差
def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

# 计算相关系数
def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0

# 输出相关系数结果
print('国民体检合格率和各区健身数目相关系数：', abs(correlation(health, Ins)))
print('国民体检合格率和各区人均健身数目相关系数：', abs(correlation(health, Ins_avg)))
print('国民体检合格率和各区人均健身面积相关系数：', abs(correlation(health, Area_avg)))
print(covariance(health, Ins), covariance(health, Ins))


###########################################################################################
# Part5: 画出国民体检合格率和各区健身设施数目、人均健身设施数目、人均健身设施面积的先验高斯分布图像
###########################################################################################
# 图像基本设置
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


