from sklearn.model_selection import train_test_split
import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.grid_search import GridSearchCV  
from sklearn import cross_validation, metrics
from sklearn.metrics import accuracy_score 
import pdb
# import matplotlib.pylab as plt  
# %matplotlib inline  
   
def percision_predict(y1,y2):
    #use FNR FPR
    assert len(y1) == len(y2)
    count = 0
    for i in range(len(y1)):
        if y1[i]==y2[i]:
            count+=1.0
    return count/len(y1)

path = '/home/wpy/dataset/data2-3.txt'
data = np.loadtxt(path, dtype = float,delimiter=None)
ds, lab = np.split(data,(-1,),axis=1)
ds_train,ds_test,lab_train,lab_test = train_test_split(ds,lab,random_state = 3,train_size = 0.8)

classes = np.unique(lab)

rf0 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,  
                                  min_samples_leaf=20,max_features=7,oob_score=True, random_state=10)  
# rf0= RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,  
#                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)  
rf0.fit(ds_train,lab_train)  

anwser_test=rf0.predict(ds_test)

percision_test=percision_predict(anwser_test,lab_test)
print(anwser_test)
dic_predict={}
for item in anwser_test:
    if item in dic_predict.keys():
        dic_predict[item]+=1
    else:
        dic_predict[item]=1
print(dic_predict)

for li in np.array(lab_test).transpose():
	print (li)
#print(np.array(lab_test).transpose())
dic_true={}
for item in li:
    if item in dic_true.keys():
        dic_true[item]+=1
    else:
        dic_true[item]=1
print(dic_true)
print(percision_predict(rf0.predict(ds_train),lab_train))
print(percision_test)

for element in np.array(lab_test).transpose():
	labels_test=element
#pdb.set_trace()
# scores=rf0.decision_function(ds_test,labels_test)

fpr,tpr,thresholds=metrics.roc_curve(lab_test,anwser_test)

# y_predprob = rf0.predict_proba(ds_test)[:,1]  
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(y,y_predprob))