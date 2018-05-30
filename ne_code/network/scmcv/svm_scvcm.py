from __future__ import division
from sklearn import svm
import numpy as np   
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn import cross_validation  
from scipy import stats
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys
import pdb
# from sklearn.decomposition import PCA
# import pdb
# import matplotlib as mpl
# from matplotlib import colors
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

path = '/home/wpy/dataset/data408.txt'
ds = np.loadtxt(path, dtype = float,delimiter=None)
data, labels= np.split(ds,(-1,),axis=1)
d=smote(ds,20)
classes=np.unique(labels)
ds, lab = np.split(d,(-1,),axis=1)
# # x_test=stats.zscore(x_test)
# #pca
# xpca = PCA(n_components=2)
# xpca.fit(ds)
# print(xpca.explained_variance_ratio_)
# new_ds = xpca.fit_transform(ds)
# #print(new_ds)
# ds = new_ds

#svm
ds_train,ds_test,lab_train,lab_test = train_test_split(ds,lab,random_state = 3,train_size = 0.8)

# With Zscore
m, n = ds_train.shape  
# pdb.set_trace()
for j in range(n):  
    features_train = ds_train[:,j]   
    features_test=ds_test[:,j]
    meanVal = features_train.mean(axis=0)  
    std = features_train.std(axis=0)  
    if std != 0:  
        ds_test[:, j] = (features_test-meanVal)/std  
    else:  
        ds_test[:, j] = 0  

ds_train=stats.zscore(ds_train)

clf = svm.SVC(C=1,kernel='rbf',decision_function_shape='ovr')
#clf = svm.SVC(decision_function_shape='ovo')
data=stats.zscore(data)
scores=cross_val_score(clf, data,labels,cv=5)
print (scores)
print (scores.mean())

clf.fit(ds_train,lab_train.ravel())
#print(clf.score(ds_train,lab_train))
lab_pre = clf.predict(ds_train)
precision1 = precision_predict(lab_pre,lab_train.ravel())
# print(lab_train.ravel(),lab_pre,precision1)
print(precision1)
lab_pre = clf.predict(ds_test)
precision2 = precision_predict(lab_pre,lab_test.ravel())

print(lab_pre)
dic_predict={}
for item in lab_pre:
    if item in dic_predict.keys():
        dic_predict[item]+=1
    else:
        dic_predict[item]=1
print(dic_predict)

print(lab_test.ravel())
#print(np.array(lab_test).transpose())
dic_true={}
for item in lab_test.ravel():
    if item in dic_true.keys():
        dic_true[item]+=1
    else:
        dic_true[item]=1
print(dic_true)
print(precision2)

#plot confusion matrix
cnf_matrix = confusion_matrix(lab_test, lab_pre)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

##############################################################
# #ROC
# prob=clf.predict_proba(ds_test)
# scores=clf.score(ds_test, lab_test)

# # Compute ROC curve and ROC area for each class  
# #plot
# # pca

# class1 = []
# class2 = []
# class3 = []
# class4 = []
# i=0
# while i<len(lab):
#     if lab[i,:]==1:
#         class1.append(ds[i,:])
#     if lab[i,:]==2:
#         class2.append(ds[i,:])
#     if lab[i,:]==3:
#         class3.append(ds[i,:])
#     if lab[i,:]==4:
#         class4.append(ds[i,:])
#     i=i+1
# x1_min, x1_max = ds[:, 0].min(), 0.5*ds[:, 0].max() 
# x2_min, x2_max = 0.5*ds[:, 1].min(), 0.5*ds[:, 1].max()  
# #pdb.set_trace()
# x1, x2 = np.mgrid[x1_min:x1_max:10j, x2_min:x2_max:10j] 
# grid_test = np.stack((x1.flat, x2.flat), axis=1) 
# cm_light = mpl.colors.ListedColormap(['#A0FFA0',  '#A0A0FF','#FFA0A0','#00FFFF'])
# cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r', 'c'])

# grid_hat = clf.predict(grid_test)
# grid_hat = grid_hat.reshape(x1.shape)
# class1=np.array(class1)
# class2=np.array(class2)
# class3=np.array(class3)
# class4=np.array(class4)	
# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
# plt.scatter(class1[:, 0], class1[:, 1], alpha=1, color='green')
# plt.scatter(class2[:, 0], class2[:, 1], alpha=1, color='blue')
# plt.scatter(class3[:, 0], class3[:, 1], alpha=1, color='red')
# plt.scatter(class3[:, 0], class3[:, 1], alpha=1, color='cyan')

# for piece in class1:
#    if piece in ds_test:
#        plt.scatter(piece[0], piece[1], alpha=0.1, color='green')
# for piece in class2:
#    if piece in ds_test:
#        plt.scatter(piece[0], piece[1], alpha=0.1, color='blue')
# for piece in class3:
#    if piece in ds_test:
#        plt.scatter(piece[0], piece[1], alpha=0.1, color='red')
# for piece in class4:
#    if piece in ds_test:
#        plt.scatter(piece[0], piece[1], alpha=0.1, color='cyan')     
               
# #plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  
# plt.xlabel('x', fontsize=13)
# plt.ylabel('y', fontsize=13)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('SVM', fontsize=15)
# # plt.grid()
# plt.show()
# # #two features,two axis
# # x_min, x_max = ds[:, 0].min(), ds[:, 0].max() 
# # y_min, y_max = ds[:, 1].min(), ds[:, 1].max()  
# # x, y = np.mgrid[x_min:x_max:150j, y_min:y_max:150j] 

# # grid_test = np.stack((x.flat, y.flat), axis=1) 

# # cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
# # #Colormap object generated from a list of colors
# # cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

# # grid_hat = clf.predict(grid_test)

# # grid_hat = grid_hat.reshape(x.shape)

