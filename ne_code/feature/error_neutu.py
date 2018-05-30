""" Extract features of swc file and make a dataset
    from pyzem.swc import swc
    tree = swc.SwcTree()
    #Load a SWC file
    tree.load('tests/data/test.swc')
    #Print the overall length of the structure
    print(tree.length())

"""
from __future__ import division  
import os
from anytree import NodeMixin, iterators, RenderTree
import math
import random
import pdb
import numpy as np
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
    if tn._type==1:
        return result
    else:
        while tn.parent._type != 1:#type 1 signifies soma
            result += tn.parent_distance()
            tn = tn.parent
            result += tn.parent_distance()
        return result

def compute_eucdist(tn):
    #compute the eucdistance between a node and soma
    result = 0
    if tn._type==1:
	    return result
    else:
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

    def node_from_id(self, nid):
        niter = iterators.PreOrderIter(self)
        for tn in niter:
            if tn.get_id() == nid:
                return tn
        return None
                      
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
    #               if not clist 
    #                   count+=1
    #               if pid == pidb:
    #                   count-=1 
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

#complex features

    def reduced_rate(self, regular = True):
        #for node
        #reduce_rate= self.radius/swctree.root.radius #self.root is a feature of a swc tree
        tn=self
        while self.parent.ntype()!=1:
            parent_tn=tn.parent
            tn=parent_tn
            
        reduce_rate=self.radius/tn.radius
        return reduce_rate
 
            
    def contraction(self, regular = True):
        #for node
        contract_rate=0.0
        tn=self
        #tn=node_from_id
        if tn.ntype()==1:
        #if the node is a soma
            return 0
        else:
            ed= compute_eucdist(tn)
            pd= compute_pathdist(tn)
            if pd==0:
                return 0
            else:
                contract_rate= ed/pd
                return contract_rate

    def diameter_rate(self, regular = True):
        #for node
        sum_rate = 0.0
        sum_radius=0.0
        pradius=self.radius()
        if len(self.children)==0:
            return 0
        else:
            niter = iterators.PreOrderIter(self)#self is a node
            for tn in niter:
                if regular:
                    if tn.is_regular():
                        sum_radius=sum_radius+tn.radius()
            sum_rate = sum_radius/pradius   
            return sum_rate/ (len(self.children))
    
    def avrg_diameter(self, regular=True):
        #for node
        sum_diameter=0.0
        avrg_diameter=0.0
        if len(self.children)==0:
            return 0
        else:
            niter = iterators.PreOrderIter(self)#self is a node
            for tn in niter:
                if regular:
                    if tn.is_regular():
                        sum_diameter=sum_diameter+2*tn.radius()
            avrg_dmt=sum_diameter/len(self.children) 
            return avrg_dmt
        
    def diameter_pow(self, regular=True):
        #for node
        pow_v=0.0
        # if len(self.children)==0:
        #     return 0
        # else:
        #     niter = iterators.PreOrderIter(self)#self is a node
        # for tn in niter:
        tn=self
        if regular:
            if tn.is_regular():
                pow_v=pow(2*self.radius(),1.5)
        return pow_v

    def Fractal_Dim(self, regular=True):
        fradim=0.0
        # niter = iterators.PreOrderIter(self)#self is a node
        # for tn in niter:
        tn=self
        if regular:
            if tn.is_regular():
                ed= compute_eucdist(self)
                pd= compute_pathdist(self)
                if pd==0:
                    return 0
                else:
                    fra_dim=(math.log(ed))/(math.log(pd))
        return fra_dim

    def Bif_tilit_local(self):
        p_distance=0.0
        c_distance=0.0
        #pdb.set_trace()
        if self.ntype()==1:
            return 0
        else:
            if len(self.children)==0:
                return 0
            else:
                pnode=self.parent
                niter = iterators.PreOrderIter(self)#self is a node
                for tn in niter:
                    cnode=tn.children[0]
                    break
            
                p_distance=self.distance(pnode)
                c_distance=self.distance(cnode)
                t_distance=pnode.distance(cnode)
                    # print(p_distance,c_distance,t_distance)
                if c_distance*p_distance!=0:
                    Bif_tilit_local=math.acos((np.square(p_distance)+np.square(c_distance)-np.square(t_distance))/(2*p_distance*c_distance))
                    return Bif_tilit_local
                else:
                    return 0

    def Bif_tilit_remote(self):
        p_distance=0.0
        c_distance=0.0
        cd_storage=[]
        if self.ntype()==1:
            return 0
        else:
            if not self.children:
                return 0
            else:
                pnode=self.parent
                niter = iterators.PreOrderIter(self)#self is a node
                for tn in niter:
                    # pdb.set_trace()
                    if len(tn.children)>1:
                        cnode=tn.parent
                        c_distance=self.distance(cnode) #find the first tip
                        p_distance=self.distance(pnode)
                        # c_distance=max(cd_storage)
                        #t_distance=pnode.distance(self.node_from_id(cd_storage.index(c_distance)))
                        t_distance=pnode.distance(cnode)
                        if c_distance*p_distance!=0:
                            Bif_tilit_remote=math.acos((np.square(p_distance)+np.square(c_distance)-np.square(t_distance))/(2*p_distance*c_distance))
                            break
                        else:
                            return 0
                    else:
                        Bif_tilit_remote=0
                return Bif_tilit_remote
    def Bif_ampl_local(self):
        p_distance=0.0
        c_distance=0.0
        count=0
        #self is a node
        if len(self.children)<2:
            return 0
        else:
            niter = iterators.PreOrderIter(self)
            for tn in niter:
                pnode=tn.parent
                if pnode==self:
                    count=count+1
                    p2_distance=tn.distance(pnode)
                    if count!=2:
                        p1_distance=p2_distance     
                        firstchild=tn
                    else:
                        secondchild=tn
                        break
            t_distance=secondchild.distance(firstchild)
            #pdb.set_trace()
            Bif_ampl_local=math.acos((np.square(p1_distance)+np.square(p2_distance)-np.square(t_distance))/(2*p1_distance*p2_distance))
            return Bif_ampl_local

    def Bif_ampl_remote(self):
        p_distance=0.0
        c_distance=0.0
        count=0
        if self._id==1:
            return 0
        else:
            pnode=self.parent
            while len(pnode.children)<2:
                pnode=pnode.parent
                if pnode.ntype()==1:
                    return 0
                    break
            #pdb.set_trace()
            for pc in pnode.children:
                niter = iterators.PreOrderIter(pc)
                for tn in niter:
                    if len(tn.children)!=1:
                        count=count+1
                        p2_distance=tn.distance(pnode)
                        if count!=2:
                            p1_distance=p2_distance
                            firstchild=tn
                        else:
                            secondchild=tn
                        break
            t_distance=secondchild.distance(firstchild)
            #pdb.set_trace()
            Bif_ampl_remote=math.acos((np.square(p1_distance)+np.square(p2_distance)-np.square(t_distance))/(2*p1_distance*p2_distance))
            return Bif_ampl_remote

    def HillmanThreshold(self,regular=True):
        count=0
        if len(self.children)<=1:
            return 0
        else:
            niter = iterators.PreOrderIter(self)#self is a node
            #pdb.set_trace()
            for tn in niter:
                if regular:
                    if tn.is_regular():
                        pnode=tn.parent
                        if pnode==self:
                            count=count+1
                            if count!=2:
                                firstchild=tn
                            else:
                                secondchild=tn
                                break
            hillman=0.5*self.radius()+0.25*firstchild.radius()+0.25*secondchild.radius()
        return hillman

    def soma_tip_distance(self):
        if self.ntype()==1:
            return 0
        else:
           #sum(allpathdist),max(allpathdist),min(allpathdist),sum(allpathdist)/len(allpathdist)])
            sumtip,maxtip,mintip,avrgtip=self.pathdistance() 
            distance=maxtip+self.pathdistance_soma()
        return distance

    # def RollerPow(self):
        
        
    def regular_surface_area(self):
        area=0.0
        area=4*math.pi*((self.radius())**2)
        return area
        
    def compute_volume(self):
        volume=0.0
        volume = (4*math.pi*(pow(self.radius(),3)))/3
        return volume
        
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
    #               if not clist 
    #                   count+=1
    #               if pid == pidb:
    #                   count-=1
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
    #                   print ("Node %d is a tip" %tn )
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
        #print(xmax,xmin)
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
        #print(xmax,xmin)
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
        #print(xmax,xmin)
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
                fobj=open('/home/wpy/dataset/error.txt','a')               
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
                #print ('Extracing the features of node')
                if random_n :
                    nodelist=[]
                    #add soma first
                    for si in range(1,5):
                        somanode=tree.node_from_id(si)
                        if somanode.ntype()==1:
                            nodelist.append(si)
                        else:
                            break
                    count_n=count-si
                    while count_n>0 :
                        newone=random.randint(si,tree.node_count())
                        if newone not in nodelist:
                            nodelist.append(newone)
                            count_n=count_n-1
                    # pdb.set_trace()
                    for snode in nodelist:
                        fobj.write(str(snode.radius())+' ')
                        fobj.write(str(snode.child_count())+' ')
                        fobj.write(str(snode.branch_count())+' ')
                        fobj.write(str(snode.bif_count())+' ')
                        fobj.write(str(snode.tip_count())+' ')
                        fobj.write(str(snode.stem_count())+' ')
             #          fobj.write(str(snode.pathdistance())+' ')
                        #fobj.write(str(snode.pathdistance_soma())+' ')                       
                        #fobj.write(str(snode.contraction())+' ')
                        fobj.write(str(snode.diameter_rate())+' ')
                        fobj.write(str(snode.diameter_pow())+' ')
                        fobj.write(str(snode.avrg_diameter())+' ')
                        #fobj.write(str(snode.Fractal_Dim())+' ')
                        #fobj.write(str(snode.Bif_tilit_local())+' ')
                        fobj.write(str(snode.Bif_tilit_remote())+' ')
                        fobj.write(str(snode.Bif_ampl_local())+' ')
                        #fobj.write(str(snode.Bif_ampl_remote())+' ')
                        fobj.write(str(snode.HillmanThreshold())+' ')
                        #fobj.write(str(snode.soma_tip_distance())+' ')
                        fobj.write(str(snode.regular_surface_area())+' ')
                        fobj.write(str(snode.compute_volume())+' ')
                        fobj.write(str(snode.ntype())+' ')
                        fobj.write('\n')                        
                else:
                    nnumber=int(input('Enter the id of your selected node:'))
                    while nnumber<=tree.node_count():
                        snode=tree.node_from_id(nnumber)
                        nnumber=nnumber+1
                        fobj.write(str(snode.radius())+' ')
                        fobj.write(str(snode.child_count())+' ')
                        fobj.write(str(snode.branch_count())+' ')
                        fobj.write(str(snode.bif_count())+' ')
                        fobj.write(str(snode.tip_count())+' ')
                        fobj.write(str(snode.stem_count())+' ')
             #          fobj.write(str(snode.pathdistance())+' ')
                        #fobj.write(str(snode.pathdistance_soma())+' ')                       
                        #fobj.write(str(snode.contraction())+' ')
                        fobj.write(str(snode.diameter_rate())+' ')
                        fobj.write(str(snode.diameter_pow())+' ')
                        fobj.write(str(snode.avrg_diameter())+' ')
                        #fobj.write(str(snode.Fractal_Dim())+' ')
                        #fobj.write(str(snode.Bif_tilit_local())+' ')
                        fobj.write(str(snode.Bif_tilit_remote())+' ')
                        fobj.write(str(snode.Bif_ampl_local())+' ')
                        #fobj.write(str(snode.Bif_ampl_remote())+' ')
                        fobj.write(str(snode.HillmanThreshold())+' ')
                        #fobj.write(str(snode.soma_tip_distance())+' ')
                        fobj.write(str(snode.regular_surface_area())+' ')
                        fobj.write(str(snode.compute_volume())+' ')
                        fobj.write(str(snode.ntype())+' ')
                        fobj.write('\n')        
            fobj.close()
    return None 

def upsampling(nodetpye):
    soma=[]
    count_soma=0
    data=[]
    labels=[]
    j=0
    # path = '/home/wpy/dataset/data2.txt'
    ds = np.loadtxt('/home/wpy/dataset/rh_up_down.txt', dtype = float,delimiter=None)
    data, labels= np.split(ds,(-1,),axis=1)
    fobj=open('/home/wpy/dataset/rh_up_down.txt','a')  

    for i in labels:
        #pdb.set_trace()
        if i==nodetpye:
            soma.append(data[count_soma])
        count_soma=count_soma+1
    while (j+1)<len(soma):
    #list(map(lambda x: x[0]+x[1], zip(soma[j], soma[j+1])))
        c=list(map(lambda x: (x[0]+x[1])/2, zip(soma[j], soma[j+1])))
        for x in c:
            fobj.write(str(x)+' ')
        fobj.write(str(nodetpye)+' ')
        fobj.write('\n')
        j=j+2
    fobj.close()
def downsampling(nodetype):
    # soma=[]
    count_soma=0
    # data=[]
    # labels=[]
    # ds = np.loadtxt('/home/wpy/dataset/rh.txt', dtype = float,delimiter=None)
    # data, labels= np.split(ds,(-1,),axis=1)
    with open("/home/wpy/dataset/rh_up_down.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
    #print(lines)
    with open("/home/wpy/dataset/rh_up_down.txt","w",encoding="utf-8") as f_w:
        for line in lines:
            if " 1 " in line:
                count_soma=count_soma+1
                if count_soma==3:
                    count_soma=0
                    continue
            f_w.write(line)
    # # path = '/home/wpy/dataset/data2.txt'

    # fobj=open('/home/wpy/dataset/rh.txt','a')  
    # for i in labels:
    #     #pdb.set_trace()
    #     if i==nodetpye:
    #         count_soma=count_soma+1
    #         if count_soma==3:
    #     fobj.write(str(nodetpye)+' ')
    #     fobj.write('\n')
    #     j=j+2
    #fobj.close()


if __name__ == '__main__':
    feature_extract('/home/wpy/dataset/error', False, False)
    # upsampling(2)
    # downsampling(1)