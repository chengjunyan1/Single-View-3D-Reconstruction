import numpy as np
import models.synnet as sn 
import models.matchnet as mn 
import models.completnet as cn 
import models.modeltools as mt
import tensorflow as tf

class SynNet:
    def __init__(self,Disp=33):
        self.vs=sn.ViewSynNet(Disp)
        self.cost=self.vs.cost
    
    def __call__(self,rgb):
        self.left=rgb
        self.right=self.vs(self.left)
        return self.right

    def vars(self):
        return self.vs.get_var()

class MatNet:
    def __init__(self,Disp=64):
        self.sm=mn.StereoMatchNet(Disp)
        self.r=self.sm.r
        self.cost=self.sm.cost
      
    def __call__(self,left,right):
        depth=self.sm(left,right)
        return depth

    def vars(self):
        return self.sm.get_var()

class SMNet:
    def __init__(self,Disp1=33,Disp2=64):
        self.syn=SynNet(Disp1)
        self.mat=MatNet(Disp2)
        self.r=self.mat.r
        self.cost=self.mat.cost
        
    def __call__(self,rgb):
        self.left=rgb
        self.right=self.syn(self.left)
        depth=self.mat(self.left,self.right)
        return depth

    def vars(self):
        vs_params=self.syn.vars()
        sm_params=self.mat.vars()
        return vs_params+sm_params

class PCNet:
    def __init__(self):
        self.pcn=cn.PointCompleNet()
        self.cost=self.pcn.cost

    def __call__(self,dep,mask):
        pcd=mt.create_pcd(dep,mask)
        self.partial=pcd
        out=self.pcn(pcd)#dep*mask)#
        return out

class SMCNet:
    def __init__(self):
        self.smn=SMNet()
        self.pcn=PCNet()
        self.cost=self.pcn.cost

    def __call__(self,left,mask):
        self.dep=self.smn(left)[self.smn.r]
        self.right=self.smn.right
        out=self.pcn(self.dep,mask)
        self.partial=self.pcn.partial
        return out
