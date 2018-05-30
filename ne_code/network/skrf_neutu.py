from __future__ import division
from sklearn import tree
from sklearn.metrics import classification_report
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
# import matplotlib.pylab as plt  
# %matplotlib inline  
from anytree import NodeMixin, iterators, RenderTree
import math

def Make_Virtual():
    return SwcNode(nid=-1)

def compute_platform_area(r1, r2, h):
    return (r1 + r2) * h * math.pi

#to test
def compute_two_node_area(tn1, tn2, remain_dist):
    """Returns the surface area formed by two nodes
    """
    r1 = tn1.radius()
    r2 = tn2.radius()
    d = tn1.distance(tn2)
    print(remain_dist)
    
    if remain_dist >= d:
        h = d
    else:
        h = remain_dist
        a = remain_dist / d
        r2 = r1 * (1 - a) + r2 * a
        
    area = compute_platform_area(r1, r2, h)
    return area

#to test
def compute_surface_area(tn, range_radius):
    area = 0
    
    #backtrace
    currentDist = 0
    parent = tn.parent
    while parent and currentDist < range_radius:
        remainDist = range_radius - currentDist                
        area += compute_two_node_area(tn, parent, remainDist)
        currentDist += tn.distance(parent)
        tn = parent
        parent = tn.parent
             
    #forwardtrace
    currentDist = 0
    childList = tn.children
    while len(childList) == 1 and currentDist < range_radius:
        child = childList[0]
        remainDist = range_radius - currentDist                
        area += compute_two_node_area(tn, child, remainDist)
        currentDist += tn.distance(child)
        tn = child
        childList = tn.children
    
    return area


class SwcNode(NodeMixin):
    """Represents a node in a SWC tree.

    A `SwcNode` object represents a node in the SWC tree structure. As defined
    in the SWC format, a node is a 3D sphere with a certain radius. It also has     an ID as its unique identifier and a type of neuronal compartment. Except
    the root node, a node should also have a parent node. To encode multiple
    SWC structures in the same SWC object, we use a negative ID to represent a
    virtual node, which does not have any geometrical meaning.

    """
    def __init__(self, nid=-1, ntype=0, radius=1, center=[0, 0, 0], parent=None):
        self._id = nid
        self._radius = radius
        self._pos = center
        self._type = ntype
        self.parent = parent

    def is_virtual(self):
        """Returns True iff the node is virtual.
        """
        return self._id < 0

    def is_regular(self):
        """Returns True iff the node is NOT virtual.
        """
        return self._id >= 0

    def get_id(self):
        """Returns the ID of the node.
        """
        return self._id

    def distance(self, tn):
        """ Returns the distance to another node.

        It returns 0 if either of the nodes is not regular.
        
        Args: 
          tn : the target node for distance measurement
        """
        if tn and self.is_regular() and tn.is_regular():
            dx = self._pos[0] - tn._pos[0]
            dy = self._pos[1] - tn._pos[1]
            dz = self._pos[2] - tn._pos[2]
            d2 = dx * dx + dy * dy + dz * dz
            
            return math.sqrt(d2)
        
        return 0.0
        
    def parent_distance(self):
        """ Returns the distance to it parent.
        """
        return self.distance(self.parent)
    
    def radius(self):
        return self._radius
    
    def scale(self, sx, sy, sz, adjusting_radius=True):
        """Transform a node by scaling
        """

        self._pos[0] *= sx
        self._pos[1] *= sy
        self._pos[2] *= sz
        
        if adjusting_radius:
            self._radius *= math.sqrt(sx * sy)
        
    def to_swc_str(self):
        return '%d %d %g %g %g %g' % (self._id, self._type, self._pos[0], self._pos[1], self._pos[2], self._radius)
    
    def get_parent_id(self):
        return -2 if self.is_root else self.parent.get_id()
                          
    def __str__(self):
        return '%d (%d): %s, %g' % (self._id, self._type, str(self._pos), self._radius)
        
class SwcTree:
    """A class for representing one or more SWC trees.

    For simplicity, we always assume that the root is a virtual node.

    """
    def __init__(self):
        self._root = Make_Virtual()

    def _print(self):
        print(RenderTree(self._root).by_attr("_id"))
        
    def clear(self):
        self._root = Make_Virtual()
        
    def is_comment(self, line):
        return line.strip().startswith('#')
        
    def root(self):
        return self._root
    
    def regular_root(self):
        return self._root.children
            
    def node_from_id(self, nid):
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if tn.get_id() == nid:
                return tn
        return None

    def parent_id(self, nid):
        tn = self.node_from_id(nid)
        if tn:
            return tn.get_parent_id()
    
    def parent_node(self, nid):
        tn = self.node_from_id(nid)
        if tn:
            return tn.parent
            
    def child_list(self, nid):
        tn = self.node_from_id(nid)
        if tn:
            return tn.children
            
    def load(self, path):
        self.clear()
        with open(path, 'r') as fp:
            lines = fp.readlines()
            nodeDict = dict()
            for line in lines:
                if not self.is_comment(line):
#                     print line
                    data = list(map(float, line.split()))
#                     print(data)
                    if len(data) == 7:
                        nid = int(data[0])
                        ntype = int(data[1])
                        pos = data[2:5]
                        radius = data[5]
                        parentId = data[6]
                        tn = SwcNode(nid=nid, ntype=ntype, radius=radius, center=pos)
                        nodeDict[nid] = (tn, parentId)
            fp.close()
            
            for _, value in nodeDict.items():
                tn = value[0]
                parentId = value[1]
                if parentId == -1:
                    tn.parent = self._root
                else:
                    parentNode = nodeDict.get(parentId)
                    if parentNode:
                        tn.parent = parentNode[0]
                        
    def save(self, path):
        with open(path, 'w') as fp:
            niter = iterators.PreOrderIter(self._root)
            for tn in niter:
                if tn.is_regular():
                    fp.write('%s %d\n' % (tn.to_swc_str(), tn.get_parent_id()))
            fp.close()
                             
    def has_regular_node(self):
        return len(self.regular_root()) > 0
    
#     def max_id(self):
#         for node in 
#         nodes = self._tree.nodes()
#         return max(nodes)
    
    def node_count(self, regular = True):
        count = 0
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    count += 1
            else:
                count += 1
                    
        return count
    
    def parent_distance(self, nid):
        d = 0
        tn = self.node(nid)
        if tn:
            parent_tn = tn.parent
            if parent_tn:
                d = tn.distance(parent_tn)
                
        return d
    
    def scale(self, sx, sy, sz, adjusting_radius=True):
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            tn.scale(sx, sy, sz, adjusting_radius)
        
    def length(self):
        niter = iterators.PreOrderIter(self._root)
        result = 0
        for tn in niter:
            result += tn.parent_distance()
                
        return result

    def radius(self, nid):
        return self.node(nid).radius()
                        

                        

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
d=smote(ds,14)

ds, lab = np.split(d,(-1,),axis=1)
ds_train=ds
lab_train=lab

classes = np.unique(lab)

rf0 = RandomForestClassifier(n_estimators= 60, max_depth=20, min_samples_split=120,  
                                min_samples_leaf=20,max_features=10 ,oob_score=True, random_state=10)  
# rf0= RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,  
#                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)  

rf0.fit(ds_train,lab_train)  

path= '/home/wpy/dataset/neutu/error.txt'
mytree = np.loadtxt(path, dtype = float,delimiter=None)
mytree_f,mytree_l=np.split(mytree,(-1,),axis=1)
mytree_answer=rf0.predict(mytree_f)

precision_test=precision_predict(mytree_answer,mytree_l)
print (precision_test)

swc = SwcTree()
swc.load('/home/wpy/dataset/error_test/09-2902-04R-01C-60x_merge_c1.Edit.swc')
i_n=1

while i_n<=swc.node_count():
    mynode=swc.node_from_id(i_n)
    mynode._type=mytree_answer[i_n-1]
    i_n=i_n+1
swc.save('/home/wpy/dataset/error_test/09-2902-04R-01C-60x_merge_c1.Edit.swc')

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


cnf_matrix = confusion_matrix(mytree_answer,mytree_l)
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
# dic_predict={}
# for item in anwser_test:
#     if item in dic_predict.keys():
#         dic_predict[item]+=1
#     else:
#         dic_predict[item]=1
# print(dic_predict)

# for li in np.array(lab_test).transpose():
# 	print (li)
# #print(np.array(lab_test).transpose())
# dic_true={}
# for item in li:
#     if item in dic_true.keys():
#         dic_true[item]+=1
#     else:
#         dic_true[item]=1
# print(dic_true)
# print(precision_predict(rf0.predict(ds_train),lab_train))
# print(precision_test)

# # y_predprob = rf0.predict_proba(ds_test)[:,1]  
# # print ("AUC Score (Train): %f" % metrics.roc_auc_score(y,y_predprob))
# cnf_matrix = confusion_matrix(lab_test, anwser_test)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=classes,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()