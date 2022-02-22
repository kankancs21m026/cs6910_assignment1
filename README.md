
## Folders

- **/fashnMnist/NeuralNetwork.py**
		Implementation of neural network. It has all important common functions for forward propagation, back propagation
		which are used by all other optimizers

- **/fashnMnist/FashnMnist.py**
		This class call NeuralNetwork or other optimizers based on input provided by used 
		
- **/fashnMnist/Initializers.py**
		This classs used to define  various weight initializers .Mostly called from NeuralNetwork
		
- **/fashnMnist/Preprocessor.py**
		This classs used to preprocess input data like apply one hot encoding,normalization etc
		
- **/fashnMnist/Activations.py**
		This classs used to define  various Activation functions and there derivatives.Mostly called from NeuralNetwork.
		
**All optimizers:**		
- fashnMnist/optimizer/Adam.py
- fashnMnist/optimizer/NAG.py
- fashnMnist/optimizer/NAdam.py
- fashnMnist/optimizer/RMSProp.py
- fashnMnist/optimizer/MomentumGradiantDecent.py
		
Please note Basic gradiant decent and stochastic gradient descent thechnique implemented in **/fashnMnist/NeuralNetwork.py**

   
   
 

