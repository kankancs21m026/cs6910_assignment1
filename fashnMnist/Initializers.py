import numpy as np
import math
class Initializers:
    def __init__(self, init='xavier'):
        self.ip=init
        
    def initialize(self,inDim,outDim=1):
        if(self.ip=='xavier'):
            return self.xavier_initializer(inDim,outDim)
        if(self.ip=='he'):
            return self.he_initializer(inDim,outDim)
        if(self.ip=='random'):
            return self.random_initializer(inDim,outDim)
        
    def xavier_initializer(self,inDim,outDim=1):
        sd = np.sqrt(6.0 / (inDim + outDim))
        lower_bnd, upper_bnd =-sd,sd
        if(outDim==1):
            return  np.random.uniform(low=lower_bnd, high=upper_bnd, size=(inDim ))
        else:
             return  np.random.uniform(low=lower_bnd, high=upper_bnd, size=(inDim, outDim))
    
    def he_initializer(self,inDim,outDim=1):
        sd = np.sqrt(2 / (inDim))
        if(outDim==1):
             return np.random.normal(0, 1, size=(inDim)) * sd
        else:      
            return np.random.normal(0, 1, size=(inDim, outDim)) * sd
            
            
    def random_initializer(self,inDim,outDim=1):
        
        if(outDim==1):
             return np.random.normal(0, 1, size=(inDim)) 
        else:
             return np.random.normal(0, 1, size=(inDim, outDim))
           
         
   
