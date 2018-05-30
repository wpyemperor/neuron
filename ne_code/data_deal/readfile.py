import numpy as np
data=[]
labels=[]
with open('/home/wpy/dataset/data1.txt','r') as f:
    for line in f:
        linelist=line.split(' ')
        data.append([float(el) for el in linelist[:-2]])
        labels.append(linelist[-2].strip())
    print (data)
    print (labels)