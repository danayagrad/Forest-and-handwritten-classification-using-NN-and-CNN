# Forest-and-handwritten-classification-using-NN-and-CNN

2 nueral network models were buit with Tensorflow to create classification models for Forest type dataset. The model aimed  to forecast forest patch's cover type according to 7 different classes. 

1.Summary: 2 models

	1.1 A fully connected neural network with four hidden layers was built. 
		- 54 inputs, four hidden layers with 50, 40, 30 and 20 neurons respectively.
		- the output layer has 7 neurons. 
		- Adam Optimizer and ReLU activation function applied to all layers.
		- A learning rate of 0.001, a batch of 100 and a dropout of 0.5 applied. 

	1.2 A fully connected neural network with 3 hidden layers.
		- 54 inputs with three hidden layers with 50, 50 and 50 neurons respectively.
		- an output layer with 7 neurons.
		- Gradient Descent Optimizer, ReLU activation function on layers.
		- Pptimal hyperparameters were found to be: learning rate = 0.05, epochs = 300, batches = 500 and dropout = 0.5.

- Both model avoided overfitting and encouraged better model generalisation by applying early stopping and dropout.
- They also have a cross entropy cost function instead of a quadratic MSE to encourage faster learning during training.
- Both used a SoftMax activation function as output layer as the classes are mutually exclusive in this dataset.

2.Algorithm: Neural Network with Tensorflow.

3.Evaluation: Prediction accuracy and confusion matrix.

4.Dataset : Forest Cover Types dataset.It contains 522,911 training examples and 58,101 testing examples. This database consisting of many 30×30m2 patches of forest from the US Forest Service (USFS) Region 2 Resource Information System.