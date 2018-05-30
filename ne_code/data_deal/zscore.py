from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import numpy as np
from scipy import stats
import pdb

def precision_predict(y1,y2):
    #use FNR FPR
    assert len(y1) == len(y2)
    count = 0
    for i in range(len(y1)):
        if y1[i]==y2[i]:
            count+=1.0
    return count/len(y1)
#读取数据
data=[]
labels=[]

path = '/home/wpy/dataset/data2-3.txt'
ds = np.loadtxt(path, dtype = float,delimiter=None)
data, labels= np.split(ds,(-1,),axis=1)

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2)
#使用信息熵作为划分标准，对决策树进行训练
x_avrg=np.mean(x_train,axis=1)
x_var=np.var(x_train,axis=1)
x_train=stats.zscore(x_train)


x_test=stats.zscore(x_test)

 m, n = x_train.shape  
    # 归一化每一个特征  
    for j in range(n):  
        features = x_train[:,j]   
        meanVal = features.mean(axis=0)  
        std = features.std(axis=0)  
        if std != 0:  
            x_test[:, j] = (features-meanVal)/std  
        else  
            x_test[:, j] = 0  
 