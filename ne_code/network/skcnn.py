# from sklearn.neural_network import MLPClassifier
# X = [[0., 0.], [1., 1.]]
# y = [0, 1]
# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
# mlp.fit(X, y)                         
# print mlp.n_layers_
# print mlp.n_iter_
# print mlp.loss_
# print mlp.out_activation_
# ####
# ####
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.cross_validation import train_test_split
import pdb
from sklearn.decomposition import PCA
data=[]
labels=[]

def precision_predict(y1,y2):
    #use FNR FPR
    assert len(y1) == len(y2)
    count = 0
    for i in range(len(y1)):
        if y1[i]==y2[i]:
            count+=1.0
    return count/len(y1)

#make dataset
# path = '/home/wpy/dataset/features_for_cai1.txt'
# ds = np.loadtxt(path, dtype = float,delimiter=None)
# data, labels= np.split(ds,(-1,),axis=1)
with open('/home/wpy/dataset/rh_up_down.txt','r') as f:
    for line in f:
        linelist=line.split(' ')
        #pdb.set_trace()
        data.append([float(el) for el in linelist[:-2]])
        labels.append(linelist[-2].strip())
#     print (data)
#     print (labels)

# print (data)
# #pdb.set_trace()
# print (labels)
#x=np.array(dat	
# x_training_data,y_training_data = training_data
# x_valid_data,y_valid_data = valid_data
# x_test_data,y_test_data = test_data


#pca
xpca = PCA(n_components=5)
xpca.fit(data)
print(xpca.explained_variance_ratio_)
new_ds = xpca.fit_transform(data)
#print(new_ds)


classes = np.unique(labels)
x_train,x_test,y_train,y_test=train_test_split(new_ds,labels,test_size=0.2)
mlp = MLPClassifier(solver='lbfgs', activation='tanh',alpha=1e-5,hidden_layer_sizes=(5,2), max_iter=200,verbose=10,random_state=1,learning_rate_init=0.001)


mlp.fit(x_train,y_train) 

anwser_train=mlp.predict(x_train)
percision_train=precision_predict(anwser_train,y_train)
anwser_test=mlp.predict(x_test)

percision_test=precision_predict(anwser_test,y_test)
print (anwser_test)
print (y_test)
print(percision_train)
#print (np.mean(anwser==y))
#test
#print (np.mean(anwser_test==y_test))
print(percision_test)
# precision,recall,thresholds=precision_recall_curve(y_train,clf.predict(x_train))
# print precision,recall,thresholds
#print (mlp.score(x_test,y_test))
# print (mlp.n_layers_)
# print (mlp.n_iter_)
# print (mlp.loss_)
# print (mlp.out_activation_)

