import numpy as np

class NeuralNet(object):

	def __init__(self, inputLayers, outputLayers, hiddenLayers, input_scaling, output_scaling):
		self.inputLayerSize = inputLayers+1
		self.outputLayerSize = outputLayers
		self.hiddenLayerSize = hiddenLayers+1
		self.performance = 0
		self.input_scaling = input_scaling
		self.output_scaling = output_scaling
		self.W1 = np.random.normal(loc =0.0, scale =1.0, size=(self.inputLayerSize,self.hiddenLayerSize))
		self.W2 = np.random.normal(loc =0.0, scale =1.0, size=(self.hiddenLayerSize,self.outputLayerSize))

	def forward(self, X):

		# Scaling inputs
		for x in range(len(X)):
			X[x] *= self.input_scaling

		# Bias input
		X.append(1)

		# Input weighting 
		self.z2 = np.dot(X, self.W1)

		# Sigmoid activation function
		self.a2 = self.sigmoid(self.z2)

		# Bias on hidden layer
		np.append(self.a2,1)

		# Output weighting 
		yHat = np.dot(self.a2, self.W2)

		# Sigmoid activation function for the output 
		yHat = self.sigmoid(yHat)

		# Scaling outputs
		yHat *= self.output_scaling

		return yHat

	def sigmoid(self, z):
		return 1/(1 + np.exp(-z))

	def perturb_weights(self, mutation_std):
		self.W1 = np.random.normal(loc =self.W1, scale =mutation_std, size=(self.inputLayerSize,self.hiddenLayerSize))
		self.W2 = np.random.normal(loc =self.W2, scale =mutation_std, size=(self.hiddenLayerSize,self.outputLayerSize))

	def store_weights(self, filename):
		np.savetxt(filename+"W1", self.W1, delimiter=',')
		np.savetxt(filename+"W2", self.W2, delimiter=',')

	def load_weights(self, filename):
		self.W1 = np.loadtxt(filename+"W1", delimiter=',')
		self.W2 = np.loadtxt(filename+"W2", delimiter=',')




"""IMPORTANT! Scale your inputs to 0-1"""
