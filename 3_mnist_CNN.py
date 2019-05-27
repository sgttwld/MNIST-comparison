"""
A mid-level implementation of a convolutional neural network 
for MNIST classification using tensorflow.keras.layers (~1.4% error)
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/classification
Date: 2019-05-25
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

## import MNIST data, normalize, and reshape
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255, x_test/255
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))

## algorithm paramters
lr = .001		# learning rate
bs = 32			# batch size
numEp = 30		# number of episodes

## placeholders for data
X = tf.placeholder(tf.float64,[None,28,28,1])
Y = tf.placeholder(tf.int64,[None])
Y_1hot = tf.one_hot(Y,10,dtype=tf.float64)

## model
C = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28,1))(X)
P = tf.keras.layers.AvgPool2D((2, 2))(C)
F = tf.keras.layers.Flatten()(P)
p = tf.keras.layers.Dense(10, activation='softmax')(F)

## objective/loss
obj = -tf.reduce_mean(Y_1hot*tf.log(p)) 	# cross entropy

## classification accuracy (for evaluation)
percent_corr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p,axis=1),Y),tf.float64))
err = 1-percent_corr

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

		

