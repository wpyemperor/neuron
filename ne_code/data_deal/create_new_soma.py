import numpy as np
import os
import pdb
# def create_new_soma():
soma=[]
count_soma=0
data=[]
labels=[]
j=0
# f=open('/home/wpy/dataset/data2.txt','r')
# for line in f:
# 	linelist=line.split(' ')
# 	data.append([float(el) for el in linelist[:-2]])
# 	labels.append(linelist[-2].strip())
path = '/home/wpy/dataset/data2.txt'
ds = np.loadtxt(path, dtype = float,delimiter=None)
data, labels= np.split(ds,(-1,),axis=1)
# print (data)
print (labels)
fobj=open('/home/wpy/dataset/data3.txt','a')  

for i in labels:
    if i==1:
        soma.append(data[count_soma])
    count_soma=count_soma+1
while (j+1)<len(soma):
#list(map(lambda x: x[0]+x[1], zip(soma[j], soma[j+1])))
    c=list(map(lambda x: (x[0]+x[1])/2, zip(soma[j], soma[j+1])))
    for x in c:
        fobj.write(str(x)+' ')
    fobj.write('\n')
    j=j+2
fobj.close()
