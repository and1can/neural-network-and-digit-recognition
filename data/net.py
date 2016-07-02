import scipy.io as sio
import random
import numpy as np
from sklearn.preprocessing import normalize

DEBUG = 0
#DEBUG  = 1
learning_rate = 0.01
stopping_criteria = 50,0000
loss = 0

train_contents = sio.loadmat('train.mat')
X = train_contents['train_images']
Y = train_contents['train_labels']
indice = np.arange(60000)
np.random.shuffle(indice)
# save the shuffled indice, so can compare if improve or not
training = indice[0:40000]   # first 50,000 to train 
indice = indice[40000:]      #truncating list
train_images = X[:,:,training]
train_label = Y[training, :]
validation = indice[0:20000]
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
stack = np.ones([40000, 1])
stackv = np.ones([20000,1])
stack = stack
if (DEBUG):
	print "shape before stack", validation_images.shape, pixel_images.shape 
	print stack.shape, "stack shape", stackv.shape
validation_images = np.hstack((validation_images, stackv))
pixel_images = np.hstack((pixel_images, stack))

if (DEBUG):
	print "validation shape after stac", validation_images.shape, pixel_images, "check if have column of 1's", pixel_images[:,784], validation_images[:,784]

#validation_images reshaped to (10,000 , 784) 
#pixel_images reshaped to (50,000 , 784
#train_label (40,000 , 1)
#valid_label (20,000 , 1)

	# issue with this is that will always output first indice that is max, can change the tie break randomly, but may not 
	# need to do so because mult. max means likely to make mistake because then choosing randomly among max


#takes in (784, ) data
# returns components for forward pass for V and W respectively (V_forward, W_forward)
def forwardPass(data, V, W):
	return "completed forward"
#takes in (784, ) data 
# returns components for backward pass for V and W respectively (V_back, W_back)
def backwardPass(data, V, W):
	return "completed backward"

# calculate mean loss 
def mean(V, W, labels):
	return "mean loss"
# calculate the other kind of loss
def cross(V, W, labels):
	return "cross loss"

def z(V, W):
	return "z_k(x)"

def trainNet(images, labels, learning_rate, loss, stop_criteria):
	loss = 1
	mu, sigma = 0, 0.01
	V = np.random.normal(mu, sigma, (200, 785))
	W = np.random.normal(mu, sigma, (10, 201))
	# add row or column of ones to V and W 
 	if (loss == 0):
 		currLoss = mean(V, W, labels)
 	else: 
 		currLoss = cross(V, W, labels)

	print "V shape", V.shape, "W shape", W.shape 
	while (currloss >= stop_criteria): 
		#pick one random data point
		data = random.choice(images) #(784, ) dimension
		forward = forwardPass(data, V, W) #returns two elements, V and W component in forward pass
		backward = backwardPass(data, V, W, forward) # returns two elements, V and W component in backward pass
		V = V - learning_rate*forward[0]*backward[0]
		W = W - learning_rate*forward[1]*backward[1]
		# calculate loss with the new updated weights
		if (loss == 0):
 			currLoss = mean(V, W, labels)
 		else: 
 			currLoss = cross(V, W, labels)
	return V, W


def predictNet(V, W, test):
	pred = []
	for t in test: 
		print(t)
		#foumula to compute result with V and W weights 
	return pred

	#compute forward prop with V and W trained weights
	#return indice that is max, if there are multiple, use some random function to get one of the tied max


trainNet(pixel_images, train_label, learning_rate, loss, stopping_criteria)
