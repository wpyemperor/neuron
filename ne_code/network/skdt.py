#class sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
#min_samples_split=2,min_samples_leaf =1, min_weight_fraction_leaf=0.0,
#max_features=None, random_state=None, max_leaf_nodes=None,class_weight=None, presort=False)
from sklearn import tree  
  
mode = tree.DecisionTreeClassifier(criterion='gini')  
  
mode.fit(X,Y)  
y_test = mode.predict(x_test)

#http://blog.csdn.net/gamer_gyt/article/details/51226904
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
x_test=
y_test = clf.predict(x_test)
clf.score(X_test,y_test) #返回在数据集X,y上的测试分数，正确率。
print ('最佳效果：%0.3f' %clf.score(X_test,y_test)) 