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
