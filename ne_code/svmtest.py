# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 20:23:34 2018

@author: wpy
""" 
'''
    SVC参数解释 
    （1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0； 
    （2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF"; 
    （3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂； 
    （4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features; 
    （5）coef0：核函数中的独立项，'RBF' and 'Poly'有效； 
    （6）probablity: 可能性估计是否使用(true or false)； 
    （7）shrinking：是否进行启发式； 
    （8）tol（default = 1e - 3）: svm结束标准的精度; 
    （9）cache_size: 制定训练所需要的内存（以MB为单位）； 
    （10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应； 
    （11）verbose: 跟多线程有关，不大明白啥意思具体； 
    （12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited; 
    （13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None 
    （14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。 
     ps：7,8,9一般不考虑。 
    '''  
from sklearn import svm
import numpy as np   
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt

def show_accuracy(y1,y2):
    assert len(y1) == len(y2)
    count = 0
    for i in range(len(y1)):
        if y1[i]==y2[i]:
            count+=1.0
    return count/len(y1)
path = 'tests/data/test3.txt'
data = np.loadtxt(path, dtype = float,delimiter=None)
x, y = np.split(data,(21,),axis=1)

xpca = PCA(n_components=2)
xpca.fit(x)
print(xpca.explained_variance_ratio_)
newx = xpca.fit_transform(x)
print(newx)
x = newx
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3,train_size = 0.6)
clf = svm.SVC(C=0.8,kernel='linear',decision_function_shape='ovr')
#clf = svm.SVC(C=0.1,kernel='rbf',gamma=0.8,decision_function_shape='ovr')
clf.fit(x_train,y_train.ravel())
print(clf.score(x_train,y_train))
y_hat = clf.predict(x_train)
precision1 = show_accuracy(y_hat,y_train.ravel())
print(y_train.ravel(),y_hat,precision1)
y_hat = clf.predict(x_test)
precision2 = show_accuracy(y_hat,y_test.ravel())
print(y_test.ravel(),y_hat,precision2)
#print('decision_function:\n', clf.decision_function(x_train))

ca1 = x[0:23,:]
ca2 = x[24:47,:]

x1_min, x1_max = x[:, 0].min(), x[:, 0].max() 
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j] 
grid_test = np.stack((x1.flat, x2.flat), axis=1) 

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
grid_hat = clf.predict(grid_test)
grid_hat = grid_hat.reshape(x1.shape)
alpha = 0.5
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.scatter(ca1[:, 0], ca1[:, 1], alpha=alpha, color='green')
plt.scatter(ca2[:, 0], ca2[:, 1], alpha=0.1, color='blue')
for ca in ca1:
    if ca in x_test:
        plt.scatter(ca[0], ca[1], alpha=0.8, color='green')
for ca in ca2:
    if ca in x_test:
        plt.scatter(ca[0], ca[1], alpha=1.0, color='blue')
#plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  
plt.xlabel('x1', fontsize=13)
plt.ylabel('x2', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('SVM', fontsize=15)
# plt.grid()
plt.show()

