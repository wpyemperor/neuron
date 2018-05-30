#-*-coding:utf-8 -*-
from __future__ import division
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import numpy as np
import pdb
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from scipy import stats
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def precision_predict(y1,y2):
    #use FNR FPR
    assert len(y1) == len(y2)
    count = 0
    for i in range(len(y1)):
        if y1[i]==y2[i]:
            count+=1.0
    return count/len(y1)
#读取数据
def smote(data, tag_index=None, max_amount=0, std_rate=2, kneighbor=2, kdistinctvalue=10, method='mean'):
    try:
        dataor=data
        data = pd.DataFrame(data)
    except:
        raise ValueError
    case_state = data.loc[:, tag_index].groupby(data.loc[:, tag_index]).count()#count the amount of every label
    case_rate = max(case_state) / min(case_state)
    location = []
    if case_rate < 2:
        print('不需要smote过程')
        return data
    else:
        # 拆分不同大小的数据集合
        less_data = np.array(
            data[data.iloc[:, tag_index] == np.array(case_state[case_state == min(case_state)].index)[0]])
        more_data = np.array(
            data[data.iloc[:, tag_index] == np.array(case_state[case_state == max(case_state)].index)[0]])
        # 找出每个少量数据中每条数据k个邻居
        neighbors = NearestNeighbors(n_neighbors=kneighbor).fit(less_data)
        for i in range(len(less_data)):
            point = less_data[i, :]
            location_set = neighbors.kneighbors([less_data[i]], return_distance=False)[0]
            location.append(location_set)
        # 确定需要将少量数据补充到上限额度
        # 判断有没有设定生成数据个数，如果没有按照std_rate(预期正负样本比)比例生成
        if max_amount > 0:
            amount = max_amount
        else:
            amount = int(max(case_state) / std_rate)
        # 初始化，判断连续还是分类变量采取不同的生成逻辑
        times = 0
        continue_index = []  # 连续变量
        class_index = []  # 分类变量
        for i in range(less_data.shape[1]):
            if len(pd.DataFrame(less_data[:, i]).drop_duplicates()) > kdistinctvalue:
                continue_index.append(i)
            else:
                class_index.append(i)
        case_update = list()
        location_transform = np.array(location)
        while times < amount:
            # 连续变量取附近k个点的重心，认为少数样本的附近也是少数样本
            new_case = []
            pool = np.random.permutation(len(location))[1]
            neighbor_group = location_transform[pool]
            if method == 'mean':
                new_case1 = less_data[list(neighbor_group), :][:, continue_index].mean(axis=0)
            # 连续样本的附近点向量上的点也是异常点
            # if method == 'random':
            #     away_index = np.random.permutation(len(neighbor_group) - 1)[1]
            #     neighbor_group_removeorigin = neighbor_group[1:][away_index]
            #     new_case1 = less_data[pool][continue_index] + np.random.rand() * (
            #         less_data[pool][continue_index] - less_data[neighbor_group_removeorigin][continue_index])
            # # 分类变量取mode
            new_case2 = np.array(pd.DataFrame(less_data[neighbor_group, :][:, class_index]).mode().iloc[0, :])
            new_case = list(new_case1) + list(new_case2)
            if times == 0:
                case_update = new_case
            else:
                case_update = np.c_[case_update, new_case]
            #print('已经生成了%s条新数据，完成百分之%.2f' % (times, times * 100 / amount))
            times = times + 1
        # less_origin_data = np.hstack((less_data[:, continue_index], less_data[:, class_index]))
        # more_origin_data = np.hstack((more_data[:, continue_index], more_data[:, class_index]))
        # data_res = np.vstack((more_origin_data, less_origin_data, np.array(case_update.T)))
        data_res = np.vstack((dataor, np.array(case_update.T)))
        # label_columns = [0] * more_origin_data.shape[0] + [1] * (
        # less_origin_data.shape[0] + np.array(case_update.T).shape[0])
        #data_res = pd.DataFrame(data_res)
    return data_res


data=[]
labels=[]

path = '/home/wpy/dataset/error.txt'
ds = np.loadtxt(path, dtype = float,delimiter=None)
# data, labels= np.split(ds,(-1,),axis=1)
d=smote(ds,16)
#d=smote(ds,20)

# path = '/home/wpy/dataset/rh_up_down.txt'
# ds = np.loadtxt(path, dtype = float,delimiter=None)
data, labels= np.split(d,(-1,),axis=1)

classes = np.unique(labels)

clf=tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
# print clf
# DecisionTreeClassifier(class_weight=None, criterion='entropy'or'gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')

#data=stats.zscore(data)

scores=cross_val_score(clf, data,labels,cv=5)

print(scores)
print(scores.mean())

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2)

clf.fit(x_train,y_train)
anwser_train=clf.predict(x_train)
# print x_train
#print (anwser_train)
#print (y_train)

percision_train=precision_predict(anwser_train,y_train)
print(percision_train)
#print (np.mean(anwser==y))
#test
anwser_test=clf.predict(x_test)
#print (anwser_test)
for lab_test in np.array(y_test).transpose():
    print (lab_test)
# print (np.array(y_test).transpose())
#print (np.mean(anwser_test==y_test))
percision_test=precision_predict(anwser_test,lab_test)
print(percision_test)

#plot confusion matrix
cnf_matrix = confusion_matrix(y_test, anwser_test)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()