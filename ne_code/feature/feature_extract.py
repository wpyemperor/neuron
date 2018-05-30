""" Extract features of swc file and make a dataset
    from pyzem.swc import swc
    tree = swc.SwcTree()
    #Load a SWC file
    tree.load('tests/data/test.swc')
    #Print the overall length of the structure
    print(tree.length())

"""

from pyzem.swc import swc
import os
#create a txt file
#def cretxt(f_name)
while True:
    fname=input('fname>')
    if os.path.exists(fname):
        print "Error:'%s' already exists" %fname
    else:
        break

fobj=open(fname,'w')
fobj.close()


tree = swc.SwcTree()
#Load a SWC file
rootdir = '\home\wpy\dataset'
list = os.listdir(rootdir) #list all file under rootdir
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path):
	    tree.load(path)
		#write features into txt file
		#open the selected file
		#fname=input('Enter filename:')
        try:
            fobj=open(fname,'a')                 # 这里的a意思是追加，这样在加了之后就不会覆盖掉源文件中的内容，如果是w则会覆盖。
        except IOError:
            print '*** file open error:'
        else:
            fobj.write('\n'+'tree.node')   #  这里的\n的意思是在源文件末尾换行，即新加内容另起一行插入。
            fobj.close()                              #   特别注意文件操作完毕后要close
        input('Press Enter to close')