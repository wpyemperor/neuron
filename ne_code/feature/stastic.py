#data stastic
from sklearn.model_selection import train_test_split
import numpy as np   
# import matplotlib.pylab as plt  
# %matplotlib inline  
   
# #导入数据，顺便看看数据的类别分布  
# train= pd.read_csv('C:\\Users\\86349\\Desktop\\train_modified\\train_modified.csv')  
# target='Disbursed' # Disbursed的值就是二元分类的输出  
# IDcol= 'ID'  
# train['Disbursed'].value_counts()  
def precision_predict(y1,y2):
    #use FNR FPR
    assert len(y1) == len(y2)
    count = 0
    for i in range(len(y1)):
        if y1[i]==y2[i]:
            count+=1.0
    return count/len(y1)
data=[]
labels=[]
# path = '/home/wpy/dataset/data2.txt'
# data = np.loadtxt(path, dtype = float,delimiter=None)
# ds, lab = np.split(data,(-1,),axis=1)
with open('/home/wpy/dataset/neutu/error.txt','r') as f:
    for line in f:
        linelist=line.split(' ')
        #pdb.set_trace()
        data.append([float(el) for el in linelist[:-2]])
        labels.append(linelist[-2].strip())
ds_train,ds_test,lab_train,lab_test = train_test_split(data,labels,random_state = 3,train_size = 0.8)
# lab.value_counts()
# lab_test.value_counts()
# lab_train.value_counts()
dic={}
for item in labels:
    if item in dic.keys():
        dic[item]+=1
    else:
        dic[item]=1
print(dic)