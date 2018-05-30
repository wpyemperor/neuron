#-*-coding:utf-8 -*-
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from scipy import stats
import pdb

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
data=[]
labels=[]

path = '/home/wpy/dataset/error.txt'
ds = np.loadtxt(path, dtype = float,delimiter=None)
data, labels= np.split(ds,(-1,),axis=1)

classes = np.unique(labels)
pdb.set_trace()
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2)


clf=tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
# print clf
# DecisionTreeClassifier(class_weight=None, criterion='entropy'or'gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')

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
print (anwser_test)
for lab_test in np.array(y_test).transpose():
	print (lab_test)
# print (np.array(y_test).transpose())
#print (np.mean(anwser_test==y_test))
percision_test=precision_predict(anwser_test,lab_test)
print(percision_test)
# recall=precision_recall_curve(y_test,anwser_test)
# print (recall
# precision, recall, thresholds = precision_recall_curve(np.array(lab_test),np.array(anwser_test))
# print(precision,recall,thresholds)
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