import scipy.io as sio
import random
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import math

DEBUG = 0
#DEBUG  = 1


train_contents = sio.loadmat('train.mat')
X = train_contents['train_images']
Y = train_contents['train_labels']
indice = np.arange(60000)
np.random.shuffle(indice)
train_num = 50000
valid_num = 10000
# save the shuffled indice, so can compare if improve or not
training = indice[0:train_num]   # first 50,000 to train 
indice = indice[train_num:]      #truncating list
train_images = X[:,:,training]
train_label = Y[training, :]
validation = indice[0:valid_num]
valid_images = X[:,:,validation]
valid_label = Y[validation, :]
pixel_images = train_images[0,:,:]
validation_images = valid_images[0,:,:]

#if (DEBUG):
	#print "before reshape"
	#print "train_images", train_images.shape, "train_label", train_label.shape, "validation", valid_images.shape


for i in range(1, 28):

    pixel_images = np.vstack([pixel_images, train_images[i,:,:]])
    validation_images = np.vstack([validation_images, valid_images[i,:,:]])
pixel_images = pixel_images.T
validation_images = validation_images.T

if (DEBUG):
	print "validation_images", validation_images.shape, "pixel_image", pixel_images.shape, "train_label", train_label.shape, "valid_label", valid_label.shape

#normalize the data and test
validation_images = normalize(validation_images)
pixel_images = normalize(pixel_images)
if (DEBUG):
	print "mean validation_images", np.mean(validation_images), "mean pixel_images", np.mean(pixel_images)
	print "now add a column or row of ones to validation", validation_images.shape, "pixel_images", pixel_images.shape

#stack = np.array([np.ones(784), 1])
stack = np.ones([train_num, 1])
stackv = np.ones([valid_num,1])
stack = stack
if (DEBUG):
	print "shape before stack", validation_images.shape, pixel_images.shape 
	print stack.shape, "stack shape", stackv.shape
validation_images = np.hstack((validation_images, stackv))
pixel_images = np.hstack((pixel_images, stack))

if (DEBUG):
	print "validation shape after stac", validation_images.shape, pixel_images, "check if have column of 1's", pixel_images[:,784], validation_images[:,784]


class Neural_Network(object):
	def __init__(self):
		#Define HyperParameters
		self.inputLayerSize = 785
		self.outputLayerSize = 10
		self.hiddenLayerSize = 200

		#Weights (parameters)

		self.W1 = np.random.normal(0, 0.01, (self.inputLayerSize, \
								 self.hiddenLayerSize))
		self.W2 = np.random.normal(0, 0.01, (self.hiddenLayerSize + 1, \
								 self.outputLayerSize))

	def sigmoid(self, z):
		#Apply sigmoid activation function
		return 1/(1+np.exp(-z))

	
	def forward(self, X):
		#Propogate inputs through network
		D = 0
		if (D):
			print "X", X.shape, "W1", self.W1.shape
		self.z2 = np.dot(X, self.W1)
		self.a2 = np.tanh(self.z2)
		if (D):
			print "a2", self.a2.shape, "W2", self.W2.shape
		self.a2 = np.hstack([self.a2, np.ones([self.a2.shape[0], 1])])
		if (D):
			print "a2 after stacking", self.a2.shape
		self.z3 = np.dot(self.a2, self.W2)
		
		if (D):
			print "z3", self.z3.shape, self.z3
		yHat = self.sigmoid(self.z3)
		if (DEBUG):
			print "yHat", yHat.shape, "X", X.shape
		return yHat



#test sigmoid
#testInput = np.arange(-6,6,0.01)
#plt.plot(testInput, sigmoid(testInput), linewidth=2)
#plt.show()

	def sigmoidPrime(self, z):
		#Derivative of Sigmoid Function
		return np.exp(-z)/((1+np.exp(-z))**2)

#test sigmoidPrime
#testValues = np.arange(-5, 5, 0.01)
#plt.plot(testValues, sigmoidPrime(testValues), linewidth=2)
#plt.show()

	def costFunctionPrime(self, X, y):
		#compute derivative with respect to W1 and W2
		#for mean loss
		# self.yHat = self.forward(X)
		# yVec = np.zeros(self.yHat.shape)
		# yVec[0,y] = 1
		

		# delta3 = np.multiply(-(yVec-self.yHat), self.sigmoidPrime(self.z3))
		# dJdW2 = np.dot(self.a2.T, delta3)

		# #delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		# if (DEBUG):
		# 	print "delta3", delta3.shape, "W2.T", self.W2.T[:,:200].shape
		# delta2 = np.dot(delta3, self.W2.T[:, :200])*(1 - np.tanh(self.z2)**2)
		# dJdW1 = np.dot(X.T, delta2)
		# return dJdW1, dJdW2 


		# # #for cross entropy
		self.yHat = self.forward(X)
		yVec = np.zeros(self.yHat.shape)
		yVec[0,y] = 1
		#print "yHat ", self.yHat, "yVec", yVec
		dJdW2 = np.multiply(-(yVec - self.yHat).T, self.a2)
		#print dJdW2.shape

	
		delta2 = np.dot(-(yVec - self.yHat), self.W2.T[:, :200])*(1 - np.tanh(self.z2)**2)
		dJdW1 = np.dot(X.T, delta2)
		return dJdW1, dJdW2.T



   	def train(self, train, label, valid, vlabel):
   		i = 0
   		stop = 1000000
   		#stop = 100
   		
   		#stop = 50000
   		#stop = 1000
   		alpha = 0.008
   		while (i < stop):
   			#data = np.array(random.choice(train))
   			r = np.random.randint(40000)
   			if (DEBUG):
   				print "train", train.shape, "label", label.shape
   			dtrain = np.array([train[r,:]])
   			dlabel = np.array([label[r, :]])
   			if (DEBUG):
   				print "dtrain", dtrain.shape, "dlabel", dlabel.shape
   			dJdW1, dJdW2 = self.costFunctionPrime(dtrain, dlabel)
   			#dJdW1, dJdW2 = self.costFunctionPrime(data, label)
			#print self.W1, self.W2, "before"
			self.W1 = self.W1 - alpha*dJdW1
			self.W2 = self.W2 - alpha*dJdW2
			#print self.W1, self.W2, "after"
			if ((i%100) == 0):
				np.savetxt("W1" + str(i), self.W1)
				np.savetxt("W2" + str(i), self.W2)
				print "save" , i, "erorr is", self.error(valid, vlabel)

			i+=1
		print "training complete"

	def error(self, train, label):
		result = self.forward(train)
		#pred = np.array((np.argmax(result, axis=1)))
		error = 0

		for i in range(label.shape[0]): 
			#print np.where(result[i,:] == max(result[i,:]))[0], result[i,:], label.shape[0]
			p = np.where(result[i,:] == max(result[i,:]))[0]
			if (len(p) > 1):
				p = p[0]
			if (p != label[i][0]):
				error += 1
			#if (DEBUG):
			#print "p is ", p, "label is ", label[i][0], "p not same as label", p != label[i][0]
		#if (DEBUG):
		print "error is", error/float(label.shape[0])

	def predict(self, train, label):
		result = self.forward(train)
		#pred = np.array((np.argmax(result, axis=1)))
		
		pred = []

		#print "result", result.shape, "result[i, :]", result[0,:], "argmax", np.argmax(result[0,:])
		#print self.a2.shape, self.a2[:,200], "self.a2"
		for i in range(label.shape[0]): 
			#print np.where(result[i,:] == max(result[i,:]))[0], result[i,:], label.shape[0]
			p = np.where(result[i,:] == max(result[i,:]))[0]
			if (len(p) > 1):
				p = p[0]
			pred.append(p)
			
			




nn = Neural_Network()
nn.train(pixel_images, train_label, validation_images, valid_label)
nn.predict(validation_images, valid_label)
#change to test data and write to a csv file
#load the weights from files

if (DEBUG):
	numgrad = computeNumericalGradient(nn, pixel_images, train_label)
	grad = nn.computeGradients(pixel_images, train_label)
	print norm(grad - numbrad)/norm(grad+numgrad)
	dJdW1, dJdW2 = nn.costFunctionPrime(pixel_images, train_label)
	print "forward", nn.forward(pixel_images).shape, "dJdW1", dJdW1.shape, "dJdW2", dJdW2.shape, "W1", nn.W1.shape, "W2", nn.W2.shape

