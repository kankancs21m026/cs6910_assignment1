# CS6910 Assignment 1 (CS21M026)
## Questions

| Questions      | Links |
| ----------- | ----------- |
| Question 1     | [Question 1](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS1.ipynb)       |
| Question 2,3,4,5,6    | [Ipython](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS2,3,4,5,6_Optimizers.ipynb)       , [Sweep1](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS4%2C5%2C6_sweepRun1.py)     ,  [Sweep2](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS4%2C5%2C6_sweepRun2.py) [Sweep3](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS4%2C5%2C6_sweepRun3.py)            |
| Question 7    | [Question 7](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS7-confusionMatrix.ipynb)       |
|Question 8| [ipython](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/Q8_CrossVsMse.ipynb)  ,[sweep](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS8_SweepRun.py)|
|Question 10| [ipython](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS10_AccurecyMnist.ipynb)|






 
 

 
## Packages

 - [fashnMnist/NeuralNetwork.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/NeuralNetwork.py)
  Implementation of neural network.It has all important common functions for forward propagation, back propagation
which are used by all other optimizers
 - [fashnMnist/FashnMnist.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/FashnMnist.py)
 This class call NeuralNetwork or other optimizers based on input provided by used 
- [fashnMnist/Initializers.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/Initializers.py)		
This classs used to define  various weight initializers .Mostly called from NeuralNetwork
- [fashnMnist/Preprocessor.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/Preprocessor.py)		
 This classs used to preprocess input data like apply one hot encoding,normalization etc
- [fashnMnist/Activations.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/Activations.py)		
 This classs used to define  various Activation functions and there derivatives.Mostly called from NeuralNetwork.
		
**All optimizers:**	
- [fashnMnist/optimizer/Adam.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/optimizer/Adam.py)
- [fashnMnist/optimizer/NAG.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/optimizer/NAG.pyy)
- [fashnMnist/optimizer/NAdam.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/optimizer/NAdam.py)
- [fashnMnist/optimizer/RMSProp.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/optimizer/RMSProp.py)
- [fashnMnist/optimizer/MomentumGradiantDecent.py](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/fashnMnist/optimizer/MomentumGradiantDecent.py)
		
Please note Basic gradiant decent and stochastic gradient descent thechnique implemented in **/fashnMnist/NeuralNetwork.py**

## Preprocess Data
```
preprocess=Preprocessor(normalization=True)
x_trainNorm, y_trainNorm, x_testNorm, y_testNorm=preprocess.Process_Fashon_mnistDataSet(x_train, y_train, x_test, y_test)
```
## Neural Network
```
FashnMnist(
	x={Features normalized using class Preprocessor}
	,y=[Training Labels.Data should be one hot encoded]
	,lr=[Learning rate ,DataType:{Float}, default:.1]
	,epochs =[Number of epochs]
	,batch=[size of batches under one epoch]
	,HiddenLayerNuron=[Hidden layer configuration: Example : [64,32,10] ] ***
	,layer1_size=[Total hidden Nurons should present in first hidden /layes , DataType{Int}]
	,layer2_size= [Total hidden Nurons should present in second hidden layes , DataType{Int}]
	,layer3_size=[Total hidden Nurons should present in Third hidden layes , DataType{Int}]
	,layer4_size=[Total hidden Nurons should present in fourth hiddenlayes , DataType{Int}]
	,layer5_size=[Total hidden Nurons should present in fifth hidden layes , DataType{Int}]
	,optimizer=['rms','adam','nadam','sgd','mgd','nag' ,default:'mgd']
	,initializer=['he','xavier','random',default: 'he']
	,activation=['tanh','sigmoid','relu' default:'tanh']
	,weight_decay=[weight decay for L2 regularization ,DataType:Float,default=0]
	,dropout_rate=[DataType:Float,default=0]
	 //Following parameters only used when use wandb log features 
	 wandbLog=[log data in wandb][values=True,False] 
         wandb=[pass wandb object]
         x_val=[Features used for cross validation during training.]
         y_val=[Labels used for cross validation during training.]
	)
***instead of using "HiddenLayerNuron" parameter user can use layer2_size,layer2_size,layer3_size,layer4_size,layer5_size to specify size of each hidden layers
```
Example 
 [checkout this notebook](https://github.com/kankancs21m026/cs6910_assignment1/blob/main/QS2,3,4,5,6_Optimizers.ipynb)
```

#setup
model=FashnMnist(
		x=x_trainNorm
		,y=y_trainNorm
		,lr=.001
		,epochs=10
		,batch=32
		,layer1_size=128
		,layer2_size=64
		,optimizer="nadam"
		,initializer="he"
		,activation="relu"
		,dropout_rate=.1
		)

# train the model
model.train() 

#GetPredicted result
pred,accTrain,lossTrain = model.GetRunResult(x_trainNorm,y_trainNorm)

```



	
