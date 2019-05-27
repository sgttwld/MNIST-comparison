"""
Simple neural network classifier for MNIST classification 
(FlatInput(28*28)-->Dense(200)-->Output(10), ~2.0% error)
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/classification
"""

import numpy as np
import tensorflow as tf
import math, os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## custom progress bar
def print_progress(i, tot, acc, acc_str, bar_length=30, wait=False):
	filled_length = int(round(bar_length * i / tot))
	# bar = '█' * filled_length + '-' * (bar_length - filled_length)
	bar = '|' * filled_length + '-' * (bar_length - filled_length)
	sys.stdout.write('\r%s/%s |%s| %s %s' % (i, tot, bar, acc_str+':', acc)),
	if i == tot-1 and not(wait):
		sys.stdout.write('\n')
	sys.stdout.flush()


## import MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255, x_test/255	# normalize

## algorithm paramters
lr = .001		# learning rate
bs = 32			# batch size
numEp = 20		# number of episodes

## model parameters
dimIN_0 = 28		# length of one side in the square image 
dimIN = dimIN_0**2  # number of input nodes
dimOUT = 10			# number of output nodes
dimH = 200			# number of hidden units

## weights and biases
W = {
	'h': tf.get_variable('hidden_weights',(dimIN,dimH), tf.float64),
	'y': tf.get_variable('output_weights',(dimH,dimOUT), tf.float64),
	}

b = {
	'h': tf.get_variable('hidden_bias', (dimH), tf.float64, tf.zeros_initializer()),
	'y': tf.get_variable('output_bias', (dimOUT), tf.float64, tf.zeros_initializer()),
	}

## placeholders for data
X = tf.placeholder(tf.float64,[None,dimIN_0,dimIN_0])
Y = tf.placeholder(tf.int64,[None])
X_flat = tf.layers.Flatten()(X)
Y_1hot = tf.one_hot(Y,dimOUT,dtype=tf.float64)

## hidden layer
H = tf.nn.sigmoid(tf.add(tf.matmul(X_flat,W['h']),b['h']))

## raw predictions
logits = tf.add(tf.matmul(H,W['y']),b['y'])

## model output
p = tf.nn.softmax(logits)

## objective/loss
obj = -tf.reduce_mean(Y_1hot*tf.log(p)) 				# cross entropy 

## classification error (for evaluation)
percent_corr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p,axis=1),Y),tf.float64))
err = 1 - percent_corr

## optimizer
optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08)
# optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08,name='Adam')
train_op = optimizer.minimize(obj)

## initializer
init = tf.global_variables_initializer()

## running the TF session
with tf.Session() as sess:

	## initializing
	sess.run(init)

	for n in range(0,numEp):
		numBatches = math.floor(len(x_train)/bs)
		t0, acc = time.time(), 0
		
		print('Ep:',n)
		for b in range(0,numBatches):
			batch_X, batch_Y = x_train[b*bs:(b+1)*bs], y_train[b*bs:(b+1)*bs]
			sess.run(train_op,feed_dict={X: batch_X, Y: batch_Y})
			acc = (b * acc + percent_corr.eval(session=sess,feed_dict={X:batch_X,Y:batch_Y}))/(b+1)
			print_progress(b, numBatches, round(acc,5), acc_str='acc', wait=True)
		T = round(time.time()-t0,2)
		acc_test = percent_corr.eval(session=sess,feed_dict={X:x_test,Y:y_test})
		sys.stdout.write(' time: %s test-acc: %s (error: %s%%)\n' % 
						(T, round(acc_test,3), round((1-acc_test)*100,3)))




