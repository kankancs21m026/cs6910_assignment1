import numpy as np
import wandb
class WanDb:
    def __init__(self,entity = "kankan-jana",project = "cs6910_assignment1"):
        self.entity =entity
        self.project = project            
    def confusionMatrix(self,predicted,true,labels):  
        wandb.login()
        wandb.init(entity=self.entity, project=self.project)
        wandb.log({"conf_mat" : wandb.sklearn.plot_confusion_matrix(true,predicted,labels)})
                
       
       
       
      
      
      
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
