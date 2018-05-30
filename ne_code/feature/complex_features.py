## complex features
import math
    def branch_count(self, regular = True):
##pid = parent_id, pidp = parent_id_before, pidpp= parent_id_before_before
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
	
	def reduced_rate(self, regular = True):
	    #for node
		#reduce_rate= self.radius/swctree.root.radius #self.root is a feature of a swc tree
        tn=self
		while get_parent_id(tn)!=1:
		    parent_tn=tn.parent
			tn=parent_tn
			
		reduce_rate=self.radius/tn.radius
		return reduce_rate
 
		    
	def contraction(self, regular = True):
	    #for node
		contract_rate=0.0
		tn= get_id(self)
		#tn=node_from_id
		if tn:
		#if the node is a soma
		    return 0
		else
			ed= compute_eucdist(tn)
		    pd= compute_pathdist(tn)
		    contract_rate= ed/pd
		    return contract_rate
	
	def diameter_rate(self, regular = True):
	    #for node
		sum_rate = 0.0
		sum_radius=0.0
		for tn in self.children
            if regular:
			    if tn.is_regular():
			        sum_radius=sum_radius+tn.radius
        sum_rate=sum_radius/self.radius		
        return float(sum_rate) / len(self.children)
	
	def avrg_diameter(self, regular=True):
        #for node
		sum_diameter=0.0
		avrg_diameter=0.0
		for tn in self.children
		    if regular:
			    if tn.is_regular():
                    sum_diameter=sum_diameter+2*tn.radius
        avrg_dmt=sum_diamter/len(self.children)	
        return avrg_dmt
		
	def diameter_pow(self, regular=True):
	    #for node
		pow=0.0
		for tn in self.children
		    if regular:
			    if tn.is_regular():
				    pow=(2*self.radius)**1.5
		return pow
		
	def Fractal_Dim(self, regular=True):
	    fradim=0.0
		for tn in self.children
		    if regular:
			    if tn.is_regular():
		            ed= compute_eucdist(self)
		            pd= compute_pathdist(self)
	                fra_dim=(math.log(ed))/(math.log(pd))
		return fra_dim
	
	def regular_surface_area(self):
	    area=0.0
		area=4*math.pi*((self.radius)**2)
		return area
		
	def compute_volume(self):
	    volume=0.0
	    volume = (4*math.pi*(self.radius**3))/3
		return volume