# coding:utf-8

'''
#Author: chenhao
#date: July.29.2018
'''

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import math
import pandas as pd

'''
###########################################################################################
# Part1: 用python中的flask测试
###########################################################################################
# Flask初始化参数尽量使用你的包名，这个初始化方式是官方推荐的
app = Flask(__name__)

@app.route('/HelloWorld')
def hello_world():
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)

# 实验结果：作为本地API，第一次使用，安装flask_restful
'''

'''
###########################################################################################
# Part2: 用python中的flask_restful测试
###########################################################################################

app = Flask(__name__)
api = Api(app)

TODOS = {
    'todo1': {'task': 'build an API'},
    'todo2': {'task': '1'},
    'todo3': {'task': 'profit!'},
}

def abort_if_todo_doesnt_exist(todo_id):
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))


parser = reqparse.RequestParser()
parser.add_argument('task')


# Todo
# shows a single todo item and lets you delete a todo item
class Todo(Resource):
    def get(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        return TODOS[todo_id]

    def delete(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        return '', 204

    def put(self, todo_id):
        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[todo_id] = task
        return task, 201


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
class TodoList(Resource):
    def get(self):
        return TODOS

    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo%i' % todo_id
        TODOS[todo_id] = {'task': args['task']}
        return TODOS[todo_id], 201


##
## Actually setup the Api resource routing here
##
api.add_resource(TodoList, '/todos')
api.add_resource(Todo, '/todos/<todo_id>')

if __name__ == '__main__':
    app.run(debug=True)
'''

'''
###########################################################################################
# Part3: 用python中的PIL库进行图像的像素访问
# 实现新吴区的住宅区、工业区和绿化区的像素提取
###########################################################################################
# 打开图像并转化为数字矩阵
img = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Logo\新吴区卫星图.png'))
img1 = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Logo\新吴区卫星图.png'))
img2 = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Logo\新吴区卫星图.png'))
img_gray = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Logo\新吴区卫星图灰度图.png'))

img_1 = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Logo\新吴区卫星图.png'))
img_2 = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Logo\新吴区卫星图.png'))
img_3 = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Logo\新吴区卫星图.png'))


# 画出原始图像
plt.figure("City_Original")
plt.imshow(img)
plt.axis('off')
plt.show()

# 查看图像信息
print(img.shape)
print(img_gray.shape)
print(img.dtype)
print(img.size)
print(type(img))
'''

'''
# 将原图像转变为灰度图，并将其保存
img_gray = img1.convert('L')
img_gray.save('新吴区卫星图灰度图.png')
'''

# 如果为RGB图像，则转换为array后，则变成一个rows*cols*channel的三维矩阵，用img[i,j,k]来访问像素值
#print(img[1,1,0])
#160, 166, 255

'''
# 在图片中点击一个点，获取其像素点的位置
imshow(img)
x = ginput(1)
print('the location of the point is: ', x)
show()
# 机场的点在图像中坐标为[(322.34848484848482, 845.65197568389067)]
# 机场的点经纬度为[(120.374096, 31.457731)]


x1 = 120.374096
y1 = 322.34848484848482
x2 = 31.457731
y2 = 845.65197568389067
# 计算对应关系 经纬度*alpha = 像素x  经纬度*belt = 像素y
# 计算得到的alpha = 2.6778891435951873
# 计算得到belt = 26.882166920554145
'''

'''
alpha = y1 / x1
belt = y2 / x2
print(alpha, belt)
'''


'''
# 画出新吴区的边界
# 提取边界的颜色
for i in range(1, 959):
    for j in range(1,1476):
        if (img[i, j, 0] == 83) & (img[i, j, 1] == 65) & (img[i, j, 2] == 253):
            img[i, j, 0] = 0
            img[i, j, 1] = 0
            img[i, j, 2] = 255
            img[i, j, 3] = 255
            print(i, j)
imshow(img)
plt.show()
'''

######################################################################################
'''
# 测试住宅区提取的效果
# 同时将灰色像素点进行高亮
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img[i, j, 0] >= 100) & (img[i, j, 0] <= 120) & (img[i, j, 1] >= 100) & (img[i, j, 1] <= 120) & (img[i, j, 2] >= 100) & (img[i, j, 2] <= 120):
            img[i, j, 0] = 255
            img[i, j, 1] = 0
            img[i, j, 2] = 0
            img[i, j, 3] = 255
            print(i, j)
imshow(img)
plt.show()
'''

'''
# 用该方法可以将边缘画成蓝色的线
# 同时标记出住宅区
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img1[i, j, 0] == 83) & (img1[i, j, 1] == 65) & (img1[i, j, 2] == 253):
            img1[i, j, 0] = 0
            img1[i, j, 1] = 0
            img1[i, j, 2] = 255
            img1[i, j, 3] = 255
            print(i, j)
        else:
            img1[i, j, 0] = 0
            img1[i, j, 1] = 0
            img1[i, j, 2] = 0
            img1[i, j, 3] = 0
#imshow(img)
#plt.show()

# 提取像素点为灰色的经纬度坐标
# 同时将灰色像素点进行高亮
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img[i, j, 0] >= 100) & (img[i, j, 0] <= 120) & (img[i, j, 1] >= 100) & (img[i, j, 1] <= 120) & (img[i, j, 2] >= 100) & (img[i, j, 2] <= 120):
            img2[i, j, 0] = 255
            img2[i, j, 1] = 0
            img2[i, j, 2] = 0
            img2[i, j, 3] = 255
            print(i, j)
        else:
            img2[i, j, 0] = 0
            img2[i, j, 1] = 0
            img2[i, j, 2] = 0
            img2[i, j, 3] = 0
#imshow(img)
#plt.show()

for i in range(1, 959):
    for j in range(1,1476):
        img[i, j, 0] = img1[i, j, 0] + img2[i, j, 0]
        img[i, j, 1] = img1[i, j, 1] + img2[i, j, 1]
        img[i, j, 2] = img1[i, j, 2] + img2[i, j, 2]
        img[i, j, 3] = img1[i, j, 3] + img2[i, j, 3]
imshow(img)
plt.show()
misc.toimage(img, cmin=0.0, cmax=...).save('卫星图_工业区.png')
'''

##########################################################################################

'''
# 测试工业区提取的效果
# 同时将蓝色像素点进行高亮
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img[i, j, 0] >= 0) & (img[i, j, 0] <= 160) & (img[i, j, 1] >= 0) & (img[i, j, 1] <= 220) & (img[i, j, 2] >= 160) & (img[i, j, 2] <= 255):
            img[i, j, 0] = 255
            img[i, j, 1] = 0
            img[i, j, 2] = 0
            img[i, j, 3] = 255
            print(i, j)
imshow(img)
plt.show()
'''

'''
# 用该方法可以将边缘画成黄色的线
# 同时标记出工业区
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img1[i, j, 0] == 83) & (img1[i, j, 1] == 65) & (img1[i, j, 2] == 253):
            img1[i, j, 0] = 0
            img1[i, j, 1] = 0
            img1[i, j, 2] = 255
            img1[i, j, 3] = 255
            print(i, j)
        else:
            img1[i, j, 0] = 0
            img1[i, j, 1] = 0
            img1[i, j, 2] = 0
            img1[i, j, 3] = 0
#imshow(img)
#plt.show()

# 提取像素点为灰色的经纬度坐标
# 同时将灰色像素点进行高亮
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img[i, j, 0] >= 0) & (img[i, j, 0] <= 160) & (img[i, j, 1] >= 0) & (img[i, j, 1] <= 220) & (img[i, j, 2] >= 160) & (img[i, j, 2] <= 255):
            img2[i, j, 0] = 255
            img2[i, j, 1] = 0
            img2[i, j, 2] = 0
            img2[i, j, 3] = 255
            print(i, j)
        else:
            img2[i, j, 0] = 0
            img2[i, j, 1] = 0
            img2[i, j, 2] = 0
            img2[i, j, 3] = 0
#imshow(img)
#plt.show()

for i in range(1, 959):
    for j in range(1,1476):
        img[i, j, 0] = img1[i, j, 0] + img2[i, j, 0]
        img[i, j, 1] = img1[i, j, 1] + img2[i, j, 1]
        img[i, j, 2] = img1[i, j, 2] + img2[i, j, 2]
        img[i, j, 3] = img1[i, j, 3] + img2[i, j, 3]
imshow(img)
plt.show()
misc.toimage(img, cmin=0.0, cmax=...).save('卫星图_工业区.png')
'''

##########################################################################################

'''
# 测试绿化区提取的效果
# 同时将绿色像素点进行高亮
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img[i, j, 0] >= 0) & (img[i, j, 0] <= 80) & (img[i, j, 1] >= 80) & (img[i, j, 1] <= 140) & (img[i, j, 2] >= 0) & (img[i, j, 2] <= 80):
            img[i, j, 0] = 255
            img[i, j, 1] = 0
            img[i, j, 2] = 0
            img[i, j, 3] = 255
            print(i, j)
imshow(img)
plt.show()
'''

'''
# 用该方法可以将边缘画成蓝色的线
# 同时标记出绿化区
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img1[i, j, 0] == 83) & (img1[i, j, 1] == 65) & (img1[i, j, 2] == 253):
            img1[i, j, 0] = 0
            img1[i, j, 1] = 0
            img1[i, j, 2] = 255
            img1[i, j, 3] = 255
            print(i, j)
        else:
            img1[i, j, 0] = 0
            img1[i, j, 1] = 0
            img1[i, j, 2] = 0
            img1[i, j, 3] = 0
#imshow(img)
#plt.show()

# 提取像素点为灰色的经纬度坐标
# 同时将灰色像素点进行高亮
for i in range(1, 959):
    for j in range(1,1476):
        #img[i, j, 3] = 0
        if (img[i, j, 0] >= 0) & (img[i, j, 0] <= 80) & (img[i, j, 1] >= 80) & (img[i, j, 1] <= 140) & (img[i, j, 2] >= 0) & (img[i, j, 2] <= 80):
            img2[i, j, 0] = 255
            img2[i, j, 1] = 0
            img2[i, j, 2] = 0
            img2[i, j, 3] = 255
            print(i, j)
        else:
            img2[i, j, 0] = 0
            img2[i, j, 1] = 0
            img2[i, j, 2] = 0
            img2[i, j, 3] = 0
#imshow(img)
#plt.show()

for i in range(1, 959):
    for j in range(1,1476):
        img[i, j, 0] = img1[i, j, 0] + img2[i, j, 0]
        img[i, j, 1] = img1[i, j, 1] + img2[i, j, 1]
        img[i, j, 2] = img1[i, j, 2] + img2[i, j, 2]
        img[i, j, 3] = img1[i, j, 3] + img2[i, j, 3]
imshow(img)
plt.show()
misc.toimage(img, cmin=0.0, cmax=...).save('卫星图_绿化区.png')
'''


'''
# 设置矩形形状，为左上角与右下角之间的点生成坐标矩阵
X, Y = np.mgrid[xmin:xmax, ymin:ymax]
# 转换为二维坐标数组
np.vstack(X.ravel(), Y.ravel())
'''

'''
# 计算方程k,b
x1 = 801.85479414202814
x2 = 502.42640692640714
y1 = 120.443784
y2 = 31.510879
x3 = 738.76185870866721
y3 = 120.430816

k1 = (y1 - y2) / (x1 - x2)
b1 = y1 - k1*x1
k2 = (y2 - y3) / (x2 - x3)
b2 = y2 - k2*x2
k3 = (y1 - y3) / (x1 - x3)
b3 = y3 - k3*x3

k = (k1 + k2 + k3) / 3
b = (b1 + b2 + b3) / 3

x4 = 679.08662613981778
y4 = 31.479783

out = k * x4 + b
distance = y4 - out
print(distance)
'''

'''
###########################################################################################
# Part4: 用python中的PIL库进行图像的像素访问
# 实现新吴区的功能区的规划
###########################################################################################

# 打开图像并转化为数字矩阵
img = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Code\Part2_Status\Figure\卫星图_总划分.png'))
img_line = np.array(Image.open('F:\天池大数据竞赛\“飞凤数创”2018全球物联网数据创新大赛\体育场地\Code\Part2_Status\Figure\卫星图_区域划分.png'))
'''

'''
##########################################################################################################
# 画出原始图像
plt.figure("City_Original")
plt.imshow(img)
plt.axis('off')
plt.show()
'''

'''
##########################################################################################################
# 查看图像信息
print(img.shape)
print(img.dtype)
print(img.size)
print(type(img))
#(959, 1476, 4)
'''


'''
##########################################################################################################
# 提取区域划分的边界
# 在图片中点击一个点，获取其像素点的位置
imshow(img_line)
x = ginput(2)
print('the location of the point is: ', x)
show()
# 线的顺序为从上到下，从左到右
# 第一条线的边界为[507, 117], [507, 483]
# 第二条线的边界为[893, 133], [893, 483]
# 第三条线的边界为[324, 483], [1230, 483]
# 第四条线的边界为[752, 483], [752, 923]
# 因此划分的结果为
# part1: x <= 507 , y <= 483
# part2: 507 < x <= 893, y <=483
# part3: x > 893, y <= 483
# part4: x <= 752, y > 483
# part5: x > 752, y >483
'''

'''
##########################################################################################################
# 统计整个新吴区的住宅区、工业区、绿化区的像素点数量
num_red = 0
num_blue = 0
num_green = 0

# 住宅区
for i in range(0, 959):
    for j in range(0, 1476):
        if (img[i, j, 0] == 255) and (img[i, j, 1] == 0) and (img[i, j, 2] == 0):
            num_red += 1
            print(num_red)

# 工业区
for i in range(0, 959):
    for j in range(0, 1476):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 0) and (img[i, j, 2] == 255):
            num_blue += 1
            print(num_blue)

# 绿化区
for i in range(0, 959):
    for j in range(0, 1476):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 255) and (img[i, j, 2] == 0):
            num_green += 1
            print(num_green)

print('the living number is', num_red)
print('the industry number is', num_blue)
print('the green number is', num_green)

# the living number is 16227
# the industry number is 20388
# the green number is 69153
'''

'''
##########################################################################################################
# 做出五个分区的住宅区、工业区、绿化区数量统计
# part1: x <= 507 , y <= 483
# part2: 507 < x <= 893, y <=483
# part3: x > 893, y <= 483
# part4: x <= 752, y > 483
# part5: x > 752, y >483

# part1
num_red = 0
num_blue = 0
num_green = 0
sum = 0

# 住宅区
for i in range(0, 483):
    for j in range(0, 507):
        if (img[i, j, 0] == 255) and (img[i, j, 1] == 0) and (img[i, j, 2] == 0):
            num_red += 1

# 工业区
for i in range(0, 483):
    for j in range(0, 507):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 0) and (img[i, j, 2] == 255):
            num_blue += 1

# 绿化区
for i in range(0, 483):
    for j in range(0, 507):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 255) and (img[i, j, 2] == 0):
            num_green += 1

sum = num_red + num_blue + num_green
num_red = num_red / sum * 100
num_blue = num_blue / sum * 100
num_green = num_green / sum * 100

print('the Part1 living number is', num_red)
print('the Part1 industry number is', num_blue)
print('the Part1 green number is', num_green)

# the Part1 living number is 5915
# the Part1 industry number is 5054
# the Part1 green number is 6807

# the Part1 living number is 33.27520252025202
# the Part1 industry number is 28.43159315931593
# the Part1 green number is 38.29320432043205

# part2
num_red = 0
num_blue = 0
num_green = 0
sum = 0

# 住宅区
for i in range(0, 483):
    for j in range(508, 893):
        if (img[i, j, 0] == 255) and (img[i, j, 1] == 0) and (img[i, j, 2] == 0):
            num_red += 1

# 工业区
for i in range(0, 483):
    for j in range(508, 893):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 0) and (img[i, j, 2] == 255):
            num_blue += 1

# 绿化区
for i in range(0, 483):
    for j in range(508, 893):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 255) and (img[i, j, 2] == 0):
            num_green += 1

sum = num_red + num_blue + num_green
num_red = num_red / sum * 100
num_blue = num_blue / sum * 100
num_green = num_green / sum * 100

print('the Part2 living number is', num_red)
print('the Part2 industry number is', num_blue)
print('the Part2 green number is', num_green)

# the Part2 living number is 3315
# the Part2 industry number is 6112
# the Part2 green number is 13100

# the Part2 living number is 14.715674523904648
# the Part2 industry number is 27.13188618102721
# the Part2 green number is 58.152439295068135

# part3
num_red = 0
num_blue = 0
num_green = 0
sum = 0

# 住宅区
for i in range(0, 483):
    for j in range(894, 1476):
        if (img[i, j, 0] == 255) and (img[i, j, 1] == 0) and (img[i, j, 2] == 0):
            num_red += 1

# 工业区
for i in range(0, 483):
    for j in range(894, 1476):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 0) and (img[i, j, 2] == 255):
            num_blue += 1

# 绿化区
for i in range(0, 483):
    for j in range(894, 1476):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 255) and (img[i, j, 2] == 0):
            num_green += 1

sum = num_red + num_blue + num_green
num_red = num_red / sum * 100
num_blue = num_blue / sum * 100
num_green = num_green / sum * 100

print('the Part3 living number is', num_red)
print('the Part3 industry number is', num_blue)
print('the Part3 green number is', num_green)

# the Part3 living number is 743
# the Part3 industry number is 807
# the Part3 green number is 15657

# the Part3 living number is 4.3180101121636545
# the Part3 industry number is 4.689951763817051
# the Part3 green number is 90.99203812401929

# part4
num_red = 0
num_blue = 0
num_green = 0
sum = 0

# 住宅区
for i in range(484, 959):
    for j in range(0, 752):
        if (img[i, j, 0] == 255) and (img[i, j, 1] == 0) and (img[i, j, 2] == 0):
            num_red += 1

# 工业区
for i in range(484, 959):
    for j in range(0, 752):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 0) and (img[i, j, 2] == 255):
            num_blue += 1

# 绿化区
for i in range(484, 959):
    for j in range(0, 752):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 255) and (img[i, j, 2] == 0):
            num_green += 1

sum = num_red + num_blue + num_green
num_red = num_red / sum * 100
num_blue = num_blue / sum * 100
num_green = num_green / sum * 100

print('the Part4 living number is', num_red)
print('the Part4 industry number is', num_blue)
print('the Part4 green number is', num_green)

# the Part4 living number is 2066
# the Part4 industry number is 3708
# the Part4 green number is 6087

# the Part4 living number is 17.418430149228563
# the Part4 industry number is 31.26211955147121
# the Part4 green number is 51.31945029930023

# part5
num_red = 0
num_blue = 0
num_green = 0
sum = 0

# 住宅区
for i in range(484, 959):
    for j in range(753, 1476):
        if (img[i, j, 0] == 255) and (img[i, j, 1] == 0) and (img[i, j, 2] == 0):
            num_red += 1

# 工业区
for i in range(484, 959):
    for j in range(753, 1476):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 0) and (img[i, j, 2] == 255):
            num_blue += 1

# 绿化区
for i in range(484, 959):
    for j in range(753, 1476):
        if (img[i, j, 0] == 0) and (img[i, j, 1] == 255) and (img[i, j, 2] == 0):
            num_green += 1

sum = num_red + num_blue + num_green
num_red = num_red / sum * 100
num_blue = num_blue / sum * 100
num_green = num_green / sum * 100

print('the Part5 living number is', num_red)
print('the Part5 industry number is', num_blue)
print('the Part5 green number is', num_green)

# the Part5 living number is 4165
# the Part5 industry number is 4695
# the Part5 green number is 27324

# the Part5 living number is 11.510612425381384
# the Part5 industry number is 12.975348220207827
# the Part5 green number is 75.51403935441078
'''

###########################################################################################
# Part5: 统计新吴区总体以及各个区的体育设施数量
###########################################################################################
# 导入数据
path = 'Data/7.Divide/'
total = pd.read_csv(path + 'Total_num.csv')

# 获得区域1的体育设施数量
row_total = total.shape[0]

dist1_total_num = 0
dist2_total_num = 0
dist3_total_num = 0
dist4_total_num = 0
dist5_total_num = 0

for i in range(row_total):
    if total.iloc[i, 2] >= 31.504066 and total.iloc[i, 3] <= 120.392782:
        dist1_total_num = dist1_total_num + 1
    if total.iloc[i, 2] >= 31.504066 and total.iloc[i, 3] >= 120.392782 and total.iloc[i, 3] <= 120.458082:
        dist2_total_num = dist2_total_num + 1
    if total.iloc[i, 2] >= 31.504066 and total.iloc[i, 3] >= 120.458082 and total.iloc[i, 3] <= 120.518582:
        dist3_total_num = dist3_total_num + 1
    if total.iloc[i, 2] <= 31.504066 and total.iloc[i, 3] <= 120.431382:
        dist4_total_num = dist4_total_num + 1
    if total.iloc[i, 2] <= 31.504066 and total.iloc[i, 3] >= 120.431382:
        dist5_total_num = dist5_total_num + 1

print(dist1_total_num)
# dist1_total_num = 109
print(dist2_total_num)
# dist2_total_num = 19
print(dist3_total_num)
# dist3_total_num = 5
print(dist4_total_num)
# dist4_total_num = 12
print(dist5_total_num)
# dist5_total_num = 34

# 计算百分比
total_num = dist1_total_num + dist2_total_num + dist3_total_num + dist4_total_num + dist5_total_num
dist1_percent = dist1_total_num / total_num * 100
dist2_percent = dist2_total_num / total_num * 100
dist3_percent = dist3_total_num / total_num * 100
dist4_percent = dist4_total_num / total_num * 100
dist5_percent = dist5_total_num / total_num * 100

print(dist1_percent, '%')
# dist1_total_num = 60.89 %
print(dist2_percent, '%')
# dist2_total_num = 10.61 %
print(dist3_percent, '%')
# dist3_total_num = 2.79 %
print(dist4_percent, '%')
# dist4_total_num = 6.70 %
print(dist5_percent, '%')
# dist5_total_num = 18.99 %