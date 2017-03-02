
"""
Created on Fri Apr 22 12:36:08 2016
@author: Krishnan Raghavan
"""
######################################################################################
# Pre Existing Libraries
from math import *
import os,sys
from sklearn.metrics import classification_report
import random
import time
import numpy as np
import itertools
import math
from collections import Counter
from numpy import genfromtxt
import heapq
from   sklearn.decomposition         import PCA
import matplotlib.pyplot                 as plt
from   matplotlib.colors             import ListedColormap
from   sklearn.cross_validation      import train_test_split
from   sklearn.preprocessing         import StandardScaler
from   sklearn.datasets              import make_moons, make_circles, make_classification
from   sklearn.neighbors             import KNeighborsClassifier
from   sklearn.svm                   import SVC
from   sklearn.tree                  import DecisionTreeClassifier
from   sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier
from   sklearn.naive_bayes           import GaussianNB
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from   sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.random_projection 		 import SparseRandomProjection
from sklearn 						 import manifold
from   sklearn                       import cluster, datasets
from   sklearn.neighbors             import kneighbors_graph
from   sklearn.preprocessing         import StandardScaler

#######################################################################################
# Libraries created by us
sys.path.append(r'C:\Users\krm9c\Dropbox\Work\Research\Common_Libraries')
path= "E:\Research_Krishnan\Data\Data-case-study-1"
from Library_Paper_two       import *
from Library_Paper_one       import import_data, traditional_MTS
#################################################################################
def RollingDataImport():
	# Start_Analysis_Bearing()
	IR    	=  np.loadtxt('IR_sample.csv', delimiter=',')
	OR    	=  np.loadtxt('OR_sample.csv', delimiter=',')
	NL    	=  np.loadtxt('NL_sample.csv', delimiter=',')
	Norm  	=  np.loadtxt('Norm.csv'     , delimiter=',')

	sheet    = 'Test';
	f        = 'IR1.xls'
	filename =  os.path.join(path,f);
	Temp_IR  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'Test';
	f        = 'OR1.xls'
	filename =  os.path.join(path,f);
	Temp_OR  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'Test';
	f        = 'NL1.xls'
	filename =  os.path.join(path,f);
	Temp_NL  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'normal';
	f        = 'Normal_1.xls'
	filename = os.path.join(path,f);
	Temp_Norm= np.array(import_data(filename,sheet, 1));


	return Temp_Norm, Temp_IR, Temp_OR, Temp_NL, IR, NL, OR, Norm
##############################################################################

from scipy.optimize import minimize
from numpy.random import rand
from numpy import *
# Basic sigmoid function for logistic regression.
def sigmoid(X):
    return 1.0 / (1.0 + math.e ** (-1.0 * X))

# Randomly initializes the weights for layer with the specified numbers of
# incoming and outgoing connections.
def randInitializeWeights(incoming, outgoing):
    epsilon_init = 0.12
    return rand(outgoing, 1 + incoming) * (2 * epsilon_init) - epsilon_init

# Adds the bias column to the matrix X.
def addBias(X):
    return np.concatenate((np.ones((X.shape[0],1)), X), 1)

# Reconstitutes the two weight matrices from a single vector, given the
# size of the input layer, the hidden layer, and the number of possible
# labels in the output.
def extractWeightMatrices(thetas, input_layer_size, hidden_layer_size, num_labels):
    theta1size = (input_layer_size + 1) * hidden_layer_size
    theta1 = reshape(thetas[:theta1size], (hidden_layer_size, input_layer_size + 1), order='A')
    theta2 = reshape(thetas[theta1size:], (num_labels, hidden_layer_size + 1), order='A')
    return theta1, theta2

# Converts single lables to one-hot vectors.
def convertLabelsToClassVectors(labels, num_classes):
    labels = labels.reshape((labels.shape[0],1))
    ycols = np.tile(labels, (1, num_classes))
    m, n = ycols.shape
    indices = np.tile(np.arange(num_classes).reshape((1,num_classes)), (m, 1))
    ymat = indices == ycols
    return ymat.astype(int)

# Returns a vector corresponding to the randomly initialized weights for the
# input layer and hidden layer.
def getInitialWeights(input_layer_size, hidden_layer_size, num_labels):
    theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    return np.append(theta1.ravel(order='A'), theta2.ravel(order='A'))

# Trains a basic multilayer perceptron. Returns weights to use for feed-forward
# pass to predict on new data.
def train(X_train, y_train, hidden_layer_size, lmda, maxIter, num_labels):
    input_layer_size = X_train.shape[1]
    initial_weights = getInitialWeights(input_layer_size, hidden_layer_size, num_labels)
    if y_train.ndim == 1:
        # Convert the labels to one-hot vectors.
        y_train = convertLabelsToClassVectors(y_train, num_labels)

    # Given weights for the input layer and hidden layer, calulates the
    # activations for the hidden layer and the output layer of a 3-layer nn.
    def getActivations(theta1, theta2):
        z2 = np.dot(addBias(X_train),theta1.T)
        a2 = np.concatenate((np.ones((z2.shape[0],1)), sigmoid(z2)), 1)
        # a2 is an m x num_hidden+1 matrix, Theta2 is a num_labels x
        # num_hidden+1 matrix
        z3 = np.dot(a2,theta2.T)
        a3 = sigmoid(z3) # Now we have an m x num_labels matrix
        return a2, a3

    # Cost function to be minimized with respect to weights.
    def costFunction(weights):
        theta1, theta2 = extractWeightMatrices(weights, input_layer_size, hidden_layer_size, num_labels)
        hidden_activation, output_activation = getActivations(theta1, theta2)
        m = X_train.shape[0]
        cost = sum((-y_train * log(output_activation)) - ((1 - y_train) * log(1-output_activation))) / m
        # Regularization
        thetasq = sum(theta1[:,1:(input_layer_size + 1)]**2) + sum(theta2[:,1:hidden_layer_size + 1]**2)
        reg = (lmda / float(2*m)) * thetasq
        print("Training loss:\t\t{:.6f}".format(cost))
        return cost + reg

    # Gradient function to pass to our optimization function.
    def calculateGradient(weights):
        theta1, theta2 = extractWeightMatrices(weights, input_layer_size, hidden_layer_size, num_labels)
        # Backpropagation - step 1: feed-forward.
        hidden_activation, output_activation = getActivations(theta1, theta2)
        m = X_train.shape[0]
        # Step 2 - the error in the output layer is just the difference
        # between the output layer and y
        delta_3 = output_activation - y_train # delta_3 is m x num_labels
        delta_3 = delta_3.T

        # Step 3
        sigmoidGrad = hidden_activation * (1 - hidden_activation)
        delta_2 = (np.dot(theta2.T,delta_3)) * sigmoidGrad.T
        delta_2 = delta_2[1:, :] # hidden_layer_size x m
        theta1_grad = np.dot(delta_2, np.concatenate((np.ones((X_train.shape[0],1)), X_train), 1))
        theta2_grad = np.dot(delta_3, hidden_activation)
        # Add regularization
        reg_grad1 = (lmda / float(m)) * theta1
        # We don't regularize the weight for the bias column
        reg_grad1[:,0] = 0
        reg_grad2 = (lmda / float(m)) * theta2;
        reg_grad2[:,0] = 0
        return np.append(ravel((theta1_grad / float(m)) + reg_grad1, order='A'), ravel((theta2_grad / float(m)) + reg_grad2, order='A'))

    # Use scipy's minimize function with method "BFGS" to find the optimum
    # weights.
    res = minimize(costFunction, initial_weights, method='BFGS', jac=calculateGradient, options={'disp': False, 'maxiter':maxIter})
    theta1, theta2 = extractWeightMatrices(res.x, input_layer_size, hidden_layer_size, num_labels)
    return theta1, theta2

# Predicts the output given input and weights.
def predict(X, theta1, theta2):
	m, n = X.shape
	X = addBias(X)
	h1 = sigmoid(np.dot(X,theta1.T))
	h2 = sigmoid(addBias(h1).dot(theta2.T))
	return h2
###########################################################################
import time
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1

# Uses Lasagne to train a multi-layer perceptron, adapted from
# http://lasagne.readthedocs.org/en/latest/user/tutorial.html
def lasagne_mlp(X_train, y_train, X_val, y_val, X_test, y_test, hidden_units=25, num_epochs=500, l2_param = 0.01, use_dropout=True):
    X_train = X_train.reshape(-1, 1, 400)
    X_val = X_val.reshape(-1, 1, 400)
    X_test = X_test.reshape(-1, 1, 400)
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')

    print("Building model and compiling functions...")
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 400),
                                     input_var=input_var)

    if use_dropout:
        # Apply 20% dropout to the input data:
        network = lasagne.layers.DropoutLayer(network, p=0.2)

    # A single hidden layer with number of hidden units as specified in the
    # parameter.
    l_hid1 = lasagne.layers.DenseLayer(
            network, num_units=hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    if use_dropout:
        # Dropout of 50%:
        l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
        # Fully-connected output layer of 10 softmax units:
        network = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    else:
        # Fully-connected output layer of 10 softmax units:
        network = lasagne.layers.DenseLayer(
            l_hid1, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # Regularization.
    l2_penalty = lasagne.regularization.regularize_layer_params_weighted({l_hid1: l2_param}, l2)
    loss = loss + l2_penalty
    # Update expressions for training, using Stochastic Gradient Descent.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Loss expression for evaluation.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # Expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # Keep track of taining and validation cost over the epochs
    epoch_cost_train = np.empty(num_epochs, dtype=float32)
    epoch_cost_val = np.empty(num_epochs, dtype=float32)
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        # We also want to keep track of the deterministic (feed-forward)
        # training error.
        train_err_ff = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            train_err_ff += err
            train_err += train_fn(inputs, targets)

            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        epoch_cost_train[epoch] = train_err_ff / train_batches
        epoch_cost_val[epoch] = val_err / val_batches
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 50, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    return epoch_cost_train, epoch_cost_val

# This function was copied verbatim from the Lasagne tutorial at
# http://lasagne.readthedocs.org/en/latest/user/tutorial.html
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
####################################
if __name__ == "__main__":
	# Full Analysis Function Calls for analyzing Rolling element Bearing Data-set
	Temp_Norm, Temp_IR, Temp_OR, Temp_NL, IR, NL, OR, Norm = RollingDataImport()
	X    =  np.concatenate((Norm, Temp_OR, Temp_IR, Temp_NL))
	Y    =  np.concatenate((np.zeros(Norm.shape[0])+0, np.zeros(Temp_OR.shape[0])+1, np.zeros(Temp_IR.shape[0])+2, np.zeros(Temp_NL.shape[0])+3))
	max = 0;
	dimension   =  3   ;
	numberFlag  = 111  ;
	par         = 1e-04 ;
	CurrentFile = []   ;
	# MD,C = initialize_calculation(Norm, X, dimension, numberFlag, CurrentFile, par);
	X_scaler = StandardScaler()
	X_scaler.fit(X)
	X 		 = X_scaler.transform(X)
	x_train, x_test, y_train, y_test = train_test_split(X, Y.astype("int0"), test_size = 0.40)
	epoch_cost_train, epoch_cost_val = lasagne_mlp(x_train, y_train, x_test, y_test, X,
 Y, hidden_units=800, num_epochs=500, l2_param=0, use_dropout=True)










	# init_weights = getInitialWeights(11, 50, 4)
	# theta1_init, theta2_init = extractWeightMatrices(init_weights,11, 50, 4)

    # pred_train = predict(X_train, theta1_init, theta2_init)
    # print sum(np.where(y_train == pred_train, 1, 0))/float(X_train.shape[0])
	# theta1, theta2 = train(X_train, y_train, 200, 0, 50,4)
	# MD, CF1= initialize_calculation(Norm, np.concatenate((Norm, Temp_OR)), dimension, numberFlag, CurrentFile, par);
	# Test = X_scaler.transform(np.concatenate((OR, IR)))
	# predictions = np.argmax(predict(np.array(Test), theta1, theta2), axis=1)
	# N_Y    =  np.concatenate((np.zeros(OR.shape[0])+1, np.zeros(IR.shape[0])+2))
	# print y_train.shape
	# print predictions.shape
	# print predictions
	# print (classification_report(N_Y.astype("int0"), predictions))
	# print ("Accuracy:", sum(np.where(N_Y == predictions, 1, 0))/float(Test.shape[0]))
	# dbn = DBN (np.array(X), Y)
	# Error_T_1(Ref, T)
	# Error_T_2(Ref, T)
	# Classification(IR, OR, NL, Norm, Temp_IR, Temp_OR, Temp_NL, Temp_Norm)
	# class_timeseries_start(Data, IR, OR, Norm, NL)
	# comparison_dimred_classification( Temp_IR, Temp_OR, Norm, Temp_Norm, Temp_NL)
	# Test_Computation()
	# Test_classification_Accuracy()
	# Some analysis on Clustering
	# Temp_Norm, Temp_IR, Temp_OR, Temp_NL, IR, NL, OR, Norm = RollingDataImport()
	# Ref  = Norm[0:100,:]
	# Data = np.concatenate((Temp_IR[0:100,:], Temp_OR[0:100,:]))
	# Y = np.concatenate(( (np.zeros((100)))+0, (np.zeros((100))+1) ))
	# Y = Y.astype(int)
	# cluster_faults(Ref, Data, Y)
