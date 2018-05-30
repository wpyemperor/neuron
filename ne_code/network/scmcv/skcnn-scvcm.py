from __future__ import division
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from scipy import stats
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys
from sklearn.ensemble import RandomForestClassifier  
from sklearn.grid_search import GridSearchCV  
from sklearn import cross_validation, metrics
from sklearn.metrics import accuracy_score 
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.cross_validation import train_test_split
import pdb
from sklearn.decomposition import PCA
# import matplotlib.pylab as plt  
# %matplotlib inline  
   
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

path = '/home/wpy/dataset/aramal.txt'
ds = np.loadtxt(path, dtype = float,delimiter=None)
# data, labels= np.split(ds,(-1,),axis=1)
d=smote(ds,20)

ds, lab = np.split(d,(-1,),axis=1)

xpca = PCA(n_components=5)
xpca.fit(ds)

new_ds = xpca.fit_transform(d)

x_train,x_test,y_train,y_test = train_test_split(ds,lab,random_state = 3,train_size = 0.8)

classes = np.unique(lab)


#print(new_ds)

mlp = MLPClassifier(solver='lbfgs', activation='tanh',alpha=1e-5,hidden_layer_sizes=(5,2), max_iter=200,verbose=10,random_state=1,learning_rate_init=0.001)


mlp.fit(x_train,y_train) 

anwser_train=mlp.predict(x_train)
percision_train=precision_predict(anwser_train,y_train)
anwser_test=mlp.predict(x_test)

percision_test=precision_predict(anwser_test,y_test)
# rf0= RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,  
#                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)  

scores=cross_val_score(mlp, ds,lab,cv=5)

print(scores)
print(scores.mean())

mlp.fit(x_train,y_train)  

anwser_test=mlp.predict(x_test)

precision_test=precision_predict(anwser_test,y_test)
print(anwser_test)
dic_predict={}
for item in anwser_test:
    if item in dic_predict.keys():
        dic_predict[item]+=1
    else:
        dic_predict[item]=1
print(dic_predict)

for li in np.array(y_test).transpose():
	print (li)
#print(np.array(lab_test).transpose())
dic_true={}
for item in li:
    if item in dic_true.keys():
        dic_true[item]+=1
    else:
        dic_true[item]=1
print(dic_true)
print(precision_predict(mlp.predict(x_train),y_train))
print(precision_test)

# y_predprob = rf0.predict_proba(ds_test)[:,1]  
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(y,y_predprob))
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