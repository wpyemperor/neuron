""" Extract features of swc file and make a dataset
    from pyzem.swc import swc
    tree = swc.SwcTree()
    #Load a SWC file
    tree.load('tests/data/test.swc')
    #Print the overall length of the structure
    print(tree.length())

"""

import os
from anytree import NodeMixin, iterators, RenderTree
import math
import random

def Make_Virtual():
    return SwcNode(nid=-1)

def compute_platform_area(r1, r2, h):
    return (r1 + r2) * h * math.pi #lateral area of circular truncated cone 

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
    #the sum of the surface area formed by the node included in range-radius
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

def compute_pathdist(tn):
    #compute the path distance between a node and soma
    result = 0
    while tn.parent._type != 1:#type 1 signifies soma
        result += tn.parent_distance()
        tn = tn.parent
    result += tn.parent_distance()
    return result

def compute_eucdist(tn):
    #compute the eucdistance between a node and soma
    result = 0
    tn1 = tn
    while tn.parent._type != 1:
        tn = tn.parent
    tn = tn.parent
    result += tn.distance(tn1)
    return result

class SwcNode(NodeMixin):
    """Represents a node in a SWC tree.

    A `SwcNode` object represents a node in the SWC tree structure. As defined
    in the SWC format, a node is a 3D sphere with a certain radius. It also has     
	an ID as its unique identifier and a type of neuronal compartment. Except
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
    #    
    def ntype(self):
        return self._type
    #
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
#    class Node_feature(node):
    def child_count(self, regular = True):
        count = 0
        niter = iterators.PreOrderIter(self)#self is a node
        for tn in niter:
            if regular:
                if tn.is_regular():
                    count += 1
            else:
                count += 1
                    
        return count
    def branch_count(self, regular = True):
##pid = parent_id, pidp = parent_id_before, pidpp= parent_id_before_before
        count = 0
        pidb = 0
        pidbb = 0
        niter = iterators.LevelOrderIter(self)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    pid = tn.get_parent_id()
                    if pid == pidb:
                        count += 1
                    if pid != pidbb:
                        count += 1
                    pidbb = pidb
                    pidb = pid
    #               print(count)
        return count	
    def bif_count(self, regular = True):
#pid = parent_id, pidp = parent_id_before, pidpp= parent_id_before_before
        count = 0
        pidbb = 0
        pidb = 0
        niter = iterators.LevelOrderIter(self)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    pid = tn.get_parent_id()
    #               print(pid,pidb,pidbb)
    #               clist=tn.child_list
    #               if not clist #若该点为叶子结点，分叉数加一
    #                   count+=1
    #               if pid == pidb:
    #                   count-=1 #若该点的父母结点与前一个点父母结点一致，则减一
    #               pidb=pid
                    if pid == pidb:
                        count += 1
                    if pid == pidbb:
                        count -= 1
                    pidbb = pidb
                    pidb = pid
    #               print(count)
        return count     

    def tip_count(self, regular = True):
        count = 0
        niter = iterators.PreOrderIter(self)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    if not tn.children:
                        count += 1
    #                   print ("Node %d is a tip" %tn )
    #   print(count)
        return count                 
    
    def stem_count(self, regular = True):
        count = 0
        niter = iterators.PreOrderIter(self)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    if tn._type ==1 :
                        count += len(tn.children)
                        if tn.parent._type ==1:
                            count -=1
        return count   


    def pathdistance(self, disttype='normal'):
        #compute the total/maximun/minimum/average of path distance
        pathdist = 0
        allpathdist = []
        alltif = []
        niter = iterators.PreOrderIter(self)
        #compute the total/maximun/minimum/average of path distance from tips to selected node
        for tn in niter:
            if tn.is_regular():
                if len(tn.children) ==0 :
                    alltif.append(tn)
        for tif in alltif:
            if tif._type != 1:
                if disttype == 'normal':
                    pathdist = compute_pathdist(tif) 
                if disttype == 'euc':
                    pathdist = compute_eucdist(tif)
                allpathdist.append(pathdist)
        return([sum(allpathdist),max(allpathdist),min(allpathdist),sum(allpathdist)/len(allpathdist)])
   
    def pathdistance_soma(self, disttype='normal'):   
                #compute the total/maximun/minimum/average of path distance
        pathdist = 0
        if disttype == 'normal':
            pathdist = compute_pathdist(self)
        if disttype == 'euc':
            pathdist = compute_eucdist(self)
        return(pathdist)


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
#class Treefeature: 
    """A class for representing one or more features of a SWC tree.

    For simplicity, we always assume that the root is a virtual node.

    """

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
    

    def branch_count(self, regular = True):
#pid = parent_id, pidp = parent_id_before, pidpp= parent_id_before_before
        count = 0
        pidb = 0
        pidbb = 0
        niter = iterators.LevelOrderIter(self._root)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    pid = tn.get_parent_id()
                    if pid == pidb:
                        count += 1
                    if pid != pidbb:
                        count += 1
                    pidbb = pidb
                    pidb = pid
    #               print(count)
        return count
  
    def bif_count(self, regular = True):
#pid = parent_id, pidp = parent_id_before, pidpp= parent_id_before_before
        count = 0
        pidbb = 0
        pidb = 0
        niter = iterators.LevelOrderIter(self._root)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    pid = tn.get_parent_id()
    #               print(pid,pidb,pidbb)
	#               clist=tn.child_list
	#               if not clist #若该点为叶子结点，分叉数加一
	#                   count+=1
	#               if pid == pidb:
    #                   count-=1 #若该点的父母结点与前一个点父母结点一致，则减一
	#               pidb=pid
                    if pid == pidb:
                        count += 1
                    if pid == pidbb:
                        count -= 1
                    pidbb = pidb
                    pidb = pid
    #               print(count)
        return count                 
    def tip_count(self, regular = True):
        count = 0
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if regular:
                if tn.is_regular():
                    if not tn.children:
                        count += 1
	#					print ("Node %d is a tip" %tn )
	#   print(count)
        return count                 
    
    def stem_count(self, regular = True):
        count = 0
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if regular:
			#   if tn.parent._type ==1:
			#       count +=1
                if tn.is_regular():
                    if tn._type ==1 :
                        count += len(tn.children)
                        if tn.parent._type ==1:
                            count -=1
        return count   

    def parent_distance(self, nid):
        d = 0
        tn = self.node(nid)
        if tn:
            parent_tn = tn.parent
            if parent_tn:
                d = tn.distance(parent_tn)
                
        return d


    def width(self,atype = None):
        xmax = xmin= 0.0
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if atype == None:
                xmax = max(xmax, tn._pos[0])
                xmin = min(xmin, tn._pos[0])
            else:
                if tn._type==atype:
                    xmax = max(xmax, tn._pos[0])
                    xmin = min(xmin, tn._pos[0])
        print(xmax,xmin)
        result = xmax - xmin
        return result
    def height(self,atype = None):
        xmax = xmin= 0.0
        niter = iterators.LevelOrderIter(self._root)
        for tn in niter:
            if atype == None:
                xmax = max(xmax, tn._pos[1])
                xmin = min(xmin, tn._pos[1])
            else:
                if tn._type==atype:
                    xmax = max(xmax, tn._pos[1])
                    xmin = min(xmin, tn._pos[1])
        print(xmax,xmin)
        result = xmax - xmin
        return result
    def depth(self,atype = None):
        xmax = xmin= 0.0
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if atype == None:
                xmax = max(xmax, tn._pos[2])
                xmin = min(xmin, tn._pos[2])
            else:
                if tn._type==atype:
                    xmax = max(xmax, tn._pos[2])
                    xmin = min(xmin, tn._pos[2])
        print(xmax,xmin)
        result = xmax - xmin
        return result

    def pathdistance(self, disttype='normal'):
        #compute the total/maximun/minimum/average of path distance
        pathdist = 0
        allpathdist = []
        alltif = []
        niter = iterators.PreOrderIter(self._root)
        for tn in niter:
            if tn.is_regular():
                if len(tn.children) ==0 :
                    alltif.append(tn)
        for tif in alltif:
            if tif._type != 1:
                if disttype == 'normal':
                    pathdist = compute_pathdist(tif) 
                if disttype == 'euc':
                    pathdist = compute_eucdist(tif)
                allpathdist.append(pathdist)
        return([sum(allpathdist),max(allpathdist),min(allpathdist),sum(allpathdist)/len(allpathdist)])
		
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
                  
"""create a txt file
#def cretxt(f_name)
# while True:
#     fname=input('fname>')
#     if os.path.exists(fname):
#         print "Error:'%s' already exists" %fname
#     else:
#         break
# fobj=open(fname,'w')
# fobj.close()
# fname=features.txt
"""
def feature_extract(rootdir, for_tree=True, random_n=True, count=0):
    tree = SwcTree()
    #Load a SWC file
    # rootdir = '/home/wpy/pyzem/tests/...'
    list = os.listdir(rootdir) #list all file under rootdir
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
	        tree.load(path)
		#write features into txt file
		#open the selected file
		#fname=input('Enter filename:')
        try:
                fobj=open('/home/wpy/dataset/features.txt','a')                 # 这里的a意思是追加，这样在加了之后就不会覆盖掉源文件中的内容，如果是w则会覆盖。
        except IOError:
                print ('*** file open error:')
        else:
                 #if choose to extract the features of selected tree
            if for_tree: 
                print ('Extracing the features of tree')
                fobj.write(str(tree.node_count())+' ')
                fobj.write(str(tree.bif_count())+' ') 
                fobj.write(str(tree.tip_count())+' ')
                fobj.write(str(tree.stem_count())+' ')
                fobj.write(str(tree.pathdistance())+' ')
                fobj.write(str(tree.width())+' ')
                fobj.write(str(tree.height())+' ')
                fobj.write(str(tree.depth())+' ')
                fobj.write(str(tree.length())+' ')
                fobj.write('\n')  
 
                    
                #otherwise extract the features of selected node of the tree
            else:
                print ('Extracing the features of node')
                if random_n :
                    nodelist=[]
                    
                    nodelist=random.sample(range(2,tree.node_count()+1),count)
                    for n in nodelist:
                        snode=tree.node_from_id(n)
                        fobj.write(str(snode.radius())+' ')
                        fobj.write(str(snode.child_count())+' ')
                        fobj.write(str(snode.branch_count())+' ')
                        fobj.write(str(snode.bif_count())+' ')
                        fobj.write(str(snode.tip_count())+' ')
                        fobj.write(str(snode.stem_count())+' ')
             #          fobj.write(str(snode.pathdistance())+' ')
                        fobj.write(str(snode.pathdistance_soma())+' ')
                        fobj.write(str(snode.ntype())+' ')
                        fobj.write('\n')                         
                else:
                    nnumber=int(input('Enter the id of your selected node:'))

                    snode=tree.node_from_id(nnumber)
                    fobj.write(str(snode.radius())+' ')
                    fobj.write(str(snode.child_count())+' ')
                    fobj.write(str(snode.branch_count())+' ')
                    fobj.write(str(snode.bif_count())+' ')
                    fobj.write(str(snode.tip_count())+' ')
                    fobj.write(str(snode.stem_count())+' ')
         #               fobj.write(str(snode.pathdistance())+' ')
                    fobj.write(str(snode.pathdistance_soma())+' ')
                    fobj.write(str(snode.ntype())+' ')
                    fobj.write('\n') 
            fobj.close()
    return None 

if __name__ == '__main__':
    feature_extract('/home/wpy/dataset/cai1/amaral/CNGversion', False, True, 20)
    