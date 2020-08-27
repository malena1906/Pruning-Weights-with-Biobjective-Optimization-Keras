'''
 Author      : Malena Reiners, M.Sc. Mathematics

 Description : Demo application of the stochastic multi gradient descent algorithm and its extension, implemented in 
               MultiobjectiveOptimizers.py (SMGD, MAdam, MRMSProp version 0.1).

               Combined for our purpose with a network pruning strategy based on l1 regularization on MNIST and CIFAR10 
               in different settings.
               Import of MultiobjectiveClasses.py, CustomLosses.py, CustomModels.py is neccessary for usage.
               Two objectives/loss functions are used to optimize the neural network in more than one goal 
               For the multi-gradient algorithm we refer to the following paper for more details:

               S. Liu and L. N. Vicente, The stochastic multi-gradient algorithm for multi-objective optimization and its
               application to supervised machine learning, ISE Technical Report 19T-011, Lehigh University.

               and for the success using different extensions and combine the MOP training with pruning weights:
               
               Reiners, M., Klamroth, K., Stiglmayr, M., 2020, Efficient and Sparse Neural Networks by Pruning 
               Weights in a Multiobjective Learning Approach, ARXIV LINK HERE

               This is the main script for demonstration purpose. Most of the plots and experimental results can be traced here.

 Input(s)    : The SMGD Algorithm to solve the multiobjective optimization problem with two loss functions, in this script is used in two 
               SGD extensions as well to demonstrate the difference between the optimizers. 

 Output(s)   : SMGD outputs a Pareto stationary point for the trade-off between both objective functions (loss functions).

               Trained convolutional neural network architecture. More details on the used architectures can be found
               in CustomModels.py.

 Notes       : The code is implemented using Python 3.7, Keras 2.3.1 and Tensorflow 1.14.0
               Please not that it is mandatory to use these versions of Tensorflow and Keras, otherwise the program cannot be executed.
               The reason for this are the changed and adapted Keras and Tensorflow functions of this particular versions.

               To extend it to multiple objectives (more than two), one may need additional packages, e.g., Gurobi, to solve a quadratic
               subproblem and compute a common descent direction of multi-objective at current point.

'''
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import backend as K
from keras import regularizers
from keras.callbacks import LearningRateScheduler, LambdaCallback, Callback

from keras.engine.training import Model 
from keras.optimizers import Optimizer, SGD, Adam, RMSprop
from keras.backend.tensorflow_backend import set_session

### custom scripts
from MultiobjectiveOptimizers import SMGD, MAdam, MRMSprop
from MultiobjectiveClasses import Multi
from CustomLosses import L1loss, L2loss, L1lossDense, L2lossConv, L1L2lossDenseConv
from CustomModels import get_data, lenet5multimodel, lenet5regmodel, vggnetmultimodel, vggnetregmodel

### for comparison reasons train on the same init weights
from tensorflow import set_random_seed
from numpy.random import seed

### Uncomment if you want to use same initialization for comparison reasons
#seed(1)
#set_random_seed(2)

### Set up GPU ### private configs
tf_config= tf.ConfigProto()
os.environ['CUDA_VISIBLE_DEVICES']= '0'
tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
allow_soft_placement=True
tf_config.gpu_options.allow_growth= True
set_session(tf.Session(config=tf_config))

### Customized inputs 
m=input("Please choose a dataset to train: mnist or cifar10? ")
print( "We will train on: " ,m)
if m == 'mnist':
    mnist=True
    cifar10=False
    opt=input("Choose: SGD or Adam or both? ")
    if opt=='SGD':
        optimizers=['multi', 'sgd']
    elif opt == 'Adam':
        optimizers=['multiadam', 'adam']
    elif opt == 'both':
        optimizers=['multiadam', 'multi', 'adam', 'sgd']
    else: 
        print("Specify an algorithm!")
elif m== 'cifar10':
    mnist=False
    cifar10=True
    opt=input("Choose: SGD or RMSProp or both? ")
    if opt=='SGD':
        optimizers=['multi', 'sgd']
    elif opt == 'RMSProp':
        optimizers=['multirms', 'rms']
    elif opt == 'both':
        optimizers=['multirms', 'multi', 'rms', 'sgd']
    else: 
        print("Specify an algorithm!")
print("We will use the following optimizers: " + str(optimizers))
pruning=input("Typ True, if you want to use pruning as well: ")
if pruning: 
    print("We will use ITP")

lr=input( "Please choose a starting learning rate: ")
learning_rate=float(lr)

print("The learning rate is" + lr )
# mnist= False
# cifar10= True
# learning_rate=float(sys.argv[1])
# pruning= False

# Define Weight Update for PRUNING
def update_weights(weight_matrix, threshold):
    sparsified_weights = []
    for w in weight_matrix:
        bool_mask = (abs(w) > threshold).astype(int)
        sparsified_weights.append(w * bool_mask)
    return sparsified_weights

class TestCallback_multi(Callback):
    def __init__(self, train_data):
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.train_data
        loss1,loss2, acc = self.model.evaluate_multi(x, y, verbose=0)
        loss1_values.append(loss1)
        loss2_values.append(loss2)
        accuracy_values.append(acc)

class TestCallback(Callback):
    def __init__(self, train_data):
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.train_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        loss_values.append(loss)
        accuracy_values.append(acc)

# Define learning rate schedule for MNIST
def lr_schedule_mnist(epoch):
    lrate = learning_rate
    lrate_e = 0.1*learning_rate
    epochs = 30
    middle = 0.75 * epochs
    frame = middle / 10
    if epoch > 0:
        lrate = -(lrate - lrate_e) * (np.exp((epoch - middle) / frame) / (
                    np.exp((epoch - middle) / frame) + 1)) + lrate
    return lrate

# Define learning rate schedule CIFAR
def lr_schedule_cifar(epoch):
    lrate = learning_rate
    lrate_e = 0.1*learning_rate
    epochs = 125
    middle = 0.75 * epochs
    frame = middle / 10
    if epoch > 0:
        lrate = -(lrate - lrate_e) * (np.exp((epoch - middle) / frame) / (
                    np.exp((epoch - middle) / frame) + 1)) + lrate
    return lrate
x_train, y_train, x_test, y_test, train_data,input_shape, epochs, num_classes = get_data(mnist, cifar10)
###################################### MNIST Training ###########################################################
if mnist: # test performance of MAdam and Adam optimizer on the same problem
    for opt in optimizers:
        # start with the same init weights
        seed(1)
        set_random_seed(2)
        # collect values while training (in evaluation mode)
        loss1_values = []
        loss2_values = []
        loss_values=[]
        val_loss_values = []
        accuracy_values = []
        val_accuracy_values = []
        nonzero_weights1 = []
        nonzero_weights2 = []
        nonzero_weights3 = []
        
        if opt == 'multiadam':
            lambdaconv = 1e-4
            model = lenet5multimodel(input_shape=input_shape, weight_decay=lambdaconv)
            model.mcompile(optimizer=MAdam(multi=True, split=False, learning_rate=learning_rate),
                           loss1='sparse_categorical_crossentropy', loss2=L1lossDense(model),
                           metrics=['accuracy'])
            nonzero_weights1.append([np.count_nonzero(
                model.get_layer('denselayer1').get_weights()[0])])
            nonzero_weights2.append([np.count_nonzero(
                model.get_layer('denselayer2').get_weights()[0])])
            nonzero_weights3.append([np.count_nonzero(
                model.get_layer('denselayer3').get_weights()[0])])

        elif opt == 'multi':
            lambdaconv = 1e-4
            momentum = 0.9
            decay_rate = learning_rate / epochs
            model = lenet5multimodel(input_shape=input_shape, weight_decay=lambdaconv)
            model.mcompile(optimizer=SMGD(multi=True, split=False, learning_rate=learning_rate, decay= decay_rate, momentum=momentum),
                           loss1='sparse_categorical_crossentropy', loss2=L1lossDense(model),
                           metrics=['accuracy'])
            nonzero_weights1.append([np.count_nonzero(
                model.get_layer('denselayer1').get_weights()[0])])
            nonzero_weights2.append([np.count_nonzero(
                model.get_layer('denselayer2').get_weights()[0])])
            nonzero_weights3.append([np.count_nonzero(
                model.get_layer('denselayer3').get_weights()[0])])

        elif opt == 'adam':
            lambdaconv=1e-4  # empirically determined
            lambdadense= 3e-4 # from first experiments (empirically determined)
            model=lenet5regmodel(lambdas=lambdadense, weight_decay=lambdaconv, input_shape= input_shape)
            model.compile(optimizer=Adam(learning_rate=learning_rate),loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

            nonzero_weights1.append([np.count_nonzero(
                                        model.get_layer('denselayer1').get_weights()[0])])
            nonzero_weights2.append([np.count_nonzero(
                                            model.get_layer('denselayer2').get_weights()[0])])
            nonzero_weights3.append([np.count_nonzero(
                                            model.get_layer('denselayer3').get_weights()[0])])
        
        elif opt == 'sgd':
            lambdaconv=1e-4 # empirically determined
            lambdadense= 3e-4 # from first experiments (empirically determined)
            momentum = 0.9
            decay_rate = learning_rate / epochs
            model=lenet5regmodel(lambdas=lambdadense, weight_decay=lambdaconv, input_shape= input_shape)
            model.compile(optimizer=SGD(learning_rate=learning_rate, decay=decay_rate, momentum=momentum),loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

            nonzero_weights1.append([np.count_nonzero(
                                        model.get_layer('denselayer1').get_weights()[0])])
            nonzero_weights2.append([np.count_nonzero(
                                            model.get_layer('denselayer2').get_weights()[0])])
            nonzero_weights3.append([np.count_nonzero(
                                            model.get_layer('denselayer3').get_weights()[0])])

        ## Define Callbacks for pruning
        threshold=0.001
        weight_callback_batch=LambdaCallback(on_batch_end= lambda batch,
                                    logs: [model.get_layer(f"{name}").set_weights(update_weights(
                                                    model.get_layer(f"{name}").get_weights(), threshold))
                                                    for name in ['denselayer1', 'denselayer2','denselayer3']]
                                    )


        safe_nonzeroweights1=LambdaCallback(on_epoch_end= lambda epoch,
                                logs: [nonzero_weights1.append([np.count_nonzero(
                                            model.get_layer('denselayer1').get_weights()[0])])
                                            ]

                                    )
        safe_nonzeroweights2=LambdaCallback(on_epoch_end= lambda epoch,
                                logs: [nonzero_weights2.append([np.count_nonzero(
                                            model.get_layer('denselayer2').get_weights()[0])])
                                            ]

                                    )
        safe_nonzeroweights3=LambdaCallback(on_epoch_end= lambda epoch,
                                logs: [nonzero_weights3.append([np.count_nonzero(
                                            model.get_layer('denselayer3').get_weights()[0])])
                                            ]

                                    )

        if pruning: 
            ## Get, Update and Set Weights before Training
            weights1 = model.get_layer('denselayer1').get_weights()
            weights2 = model.get_layer('denselayer2').get_weights()
            weights3 = model.get_layer('denselayer3').get_weights() 

            sparsified_weights1 = update_weights(weights1, threshold)
            sparsified_weights2 = update_weights(weights2, threshold)
            sparsified_weights3 = update_weights(weights3, threshold)

            model.get_layer('denselayer1').set_weights(sparsified_weights1)
            model.get_layer('denselayer2').set_weights(sparsified_weights2)
            model.get_layer('denselayer3').set_weights(sparsified_weights3)

            nonzero_weights1.append([np.count_nonzero(
                                        model.get_layer('denselayer1').get_weights()[0])])
            nonzero_weights2.append([np.count_nonzero(
                                            model.get_layer('denselayer2').get_weights()[0])])
            nonzero_weights3.append([np.count_nonzero(
                                            model.get_layer('denselayer3').get_weights()[0])])



        ## Start Training  
        if opt == 'multiadam': 
            if pruning: 
                history_multiadam=model.mfit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_mnist),weight_callback_batch,safe_nonzeroweights1, safe_nonzeroweights2, safe_nonzeroweights3,TestCallback_multi((x_train,y_train))])
                nonzero_weights1.append([np.count_nonzero(
                                    model.get_layer('denselayer1').get_weights()[0])])
                nonzero_weights2.append([np.count_nonzero(
                                            model.get_layer('denselayer2').get_weights()[0])])
                nonzero_weights3.append([np.count_nonzero(
                                            model.get_layer('denselayer3').get_weights()[0])])
                nonzero_weights1_multiadam= nonzero_weights1
                nonzero_weights2_multiadam= nonzero_weights2
                nonzero_weights3_multiadam= nonzero_weights3
            else:    
                history_multiadam=model.mfit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_mnist),TestCallback_multi((x_train,y_train))])
            accuracy_values_multiadam= accuracy_values
            accuracy_values=[]
            loss1_values_multiadam = loss1_values
            loss1_values = []
            loss2_values_multiadam = loss2_values
            loss2_values = []

            weights1 = model.get_layer("denselayer1").get_weights() #weights and biases of the layer
            L1w1=sum(sum(sum(np.abs(weights1))))
            L0w1=np.count_nonzero(weights1[0]) + np.count_nonzero(weights1[1])

            weights2 = model.get_layer("denselayer2").get_weights() 
            L1w2=sum(sum(sum(np.abs(weights2))))
            L0w2= np.count_nonzero(weights2[0]) + np.count_nonzero(weights2[1])
            
            weights3 = model.get_layer("denselayer3").get_weights()
            L1w3=sum(sum(sum(np.abs(weights3))))
            L0w3=np.count_nonzero(weights3[0]) + np.count_nonzero(weights3[1])

            L0ges_multiadam=L0w1+L0w2+L0w3
            L1ges_multiadam= L1w1+L1w2+L1w3
            L0multiadam= [L0w1, L0w2, L0w3, L0ges_multiadam]
            L1multiadam= [L1w1, L1w2, L1w3, L1ges_multiadam]
            
            # EVALUATE
            [train_loss1_multiadam, train_loss2_multiadam, train_accuracy_multiadam] = model.evaluate_multi(x_train, y_train)
            [test_loss1_multiadam, test_loss2_multiadam, test_accuracy_multiadam] = model.evaluate_multi(x_test, y_test)
        
        elif opt == 'multi': 
            if pruning: 
                history_multi=model.mfit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_mnist), weight_callback_batch,safe_nonzeroweights1, safe_nonzeroweights2, safe_nonzeroweights3,TestCallback_multi((x_train,y_train))])
                nonzero_weights1.append([np.count_nonzero(
                                    model.get_layer('denselayer1').get_weights()[0])])
                nonzero_weights2.append([np.count_nonzero(
                                            model.get_layer('denselayer2').get_weights()[0])])
                nonzero_weights3.append([np.count_nonzero(
                                            model.get_layer('denselayer3').get_weights()[0])])
                nonzero_weights1_multi= nonzero_weights1
                nonzero_weights2_multi= nonzero_weights2
                nonzero_weights3_multi= nonzero_weights3
            else:    
                history_multi=model.mfit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_mnist),TestCallback_multi((x_train,y_train))])
            accuracy_values_multi= accuracy_values
            accuracy_values=[]
            loss1_values_multi = loss1_values
            loss1_values = []
            loss2_values_multi = loss2_values
            loss2_values = []

            weights1 = model.get_layer("denselayer1").get_weights() #weights and biases of the layer
            L1w1=sum(sum(sum(np.abs(weights1))))
            L0w1=np.count_nonzero(weights1[0]) + np.count_nonzero(weights1[1])

            weights2 = model.get_layer("denselayer2").get_weights() 
            L1w2=sum(sum(sum(np.abs(weights2))))
            L0w2= np.count_nonzero(weights2[0]) + np.count_nonzero(weights2[1])
            
            weights3 = model.get_layer("denselayer3").get_weights()
            L1w3=sum(sum(sum(np.abs(weights3))))
            L0w3=np.count_nonzero(weights3[0]) + np.count_nonzero(weights3[1])

            L0ges_multi=L0w1+L0w2+L0w3
            L1ges_multi= L1w1+L1w2+L1w3
            L0multi= [L0w1, L0w2, L0w3, L0ges_multi]
            L1multi= [L1w1, L1w2, L1w3, L1ges_multi]
            
            # EVALUATE
            [train_loss1_multi, train_loss2_multi, train_accuracy_multi] = model.evaluate_multi(x_train, y_train)
            [test_loss1_multi, test_loss2_multi, test_accuracy_multi] = model.evaluate_multi(x_test, y_test)


        elif opt == 'adam':
            if pruning: 
                history_adam= model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_mnist),weight_callback_batch,safe_nonzeroweights1, safe_nonzeroweights2, safe_nonzeroweights3,TestCallback((x_train,y_train))])         
                nonzero_weights1.append([np.count_nonzero(
                                        model.get_layer('denselayer1').get_weights()[0])])
                nonzero_weights2.append([np.count_nonzero(
                                            model.get_layer('denselayer2').get_weights()[0])])
                nonzero_weights3.append([np.count_nonzero(
                                            model.get_layer('denselayer3').get_weights()[0])])
                nonzero_weights1_adam= nonzero_weights1
                nonzero_weights2_adam= nonzero_weights2
                nonzero_weights3_adam= nonzero_weights3
            else: 
                history_adam= model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_mnist),TestCallback((x_train,y_train))])
            accuracy_values_adam= accuracy_values
            accuracy_values=[]
            loss_values_adam = loss_values
            loss_values = []
            weights1 = model.get_layer("denselayer1").get_weights() #weights and biases of the layer
            L1w1=sum(sum(sum(np.abs(weights1))))
            L0w1=np.count_nonzero(weights1[0]) + np.count_nonzero(weights1[1])

            weights2 = model.get_layer("denselayer2").get_weights() 
            L1w2=sum(sum(sum(np.abs(weights2))))
            L0w2= np.count_nonzero(weights2[0]) + np.count_nonzero(weights2[1])
            
            weights3 = model.get_layer("denselayer3").get_weights()
            L1w3=sum(sum(sum(np.abs(weights3))))
            L0w3=np.count_nonzero(weights3[0]) + np.count_nonzero(weights3[1])

            L0ges_adam= L0w1+L0w2+L0w3
            L1ges_adam= L1w1+L1w2+L1w3
            L0_adam= [L0w1, L0w2, L0w3,L0ges_adam]
            L1_adam= [L1w1, L1w2, L1w3, L1ges_adam]
            # EVALUATE
            [train_loss_adam, train_accuracy_adam]= model.evaluate(x_train, y_train)
            [test_loss_adam, test_accuracy_adam] = model.evaluate(x_test, y_test)
        
        elif opt == 'sgd':
            if pruning: 
                history_sgd= model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_mnist),weight_callback_batch,safe_nonzeroweights1, safe_nonzeroweights2, safe_nonzeroweights3,TestCallback((x_train,y_train))])         
                nonzero_weights1.append([np.count_nonzero(
                                        model.get_layer('denselayer1').get_weights()[0])])
                nonzero_weights2.append([np.count_nonzero(
                                            model.get_layer('denselayer2').get_weights()[0])])
                nonzero_weights3.append([np.count_nonzero(
                                            model.get_layer('denselayer3').get_weights()[0])])
                nonzero_weights1_sgd= nonzero_weights1
                nonzero_weights2_sgd= nonzero_weights2
                nonzero_weights3_sgd= nonzero_weights3
            else: 
                history_sgd= model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_mnist),TestCallback((x_train,y_train))])
            accuracy_values_sgd= accuracy_values
            accuracy_values=[]
            loss_values_sgd = loss_values
            loss_values = []
            weights1 = model.get_layer("denselayer1").get_weights() #weights and biases of the layer
            L1w1=sum(sum(sum(np.abs(weights1))))
            L0w1=np.count_nonzero(weights1[0]) + np.count_nonzero(weights1[1])

            weights2 = model.get_layer("denselayer2").get_weights() 
            L1w2=sum(sum(sum(np.abs(weights2))))
            L0w2= np.count_nonzero(weights2[0]) + np.count_nonzero(weights2[1])
            
            weights3 = model.get_layer("denselayer3").get_weights()
            L1w3=sum(sum(sum(np.abs(weights3))))
            L0w3=np.count_nonzero(weights3[0]) + np.count_nonzero(weights3[1])

            L0ges_sgd= L0w1+L0w2+L0w3
            L1ges_sgd= L1w1+L1w2+L1w3
            L0_sgd= [L0w1, L0w2, L0w3,L0ges_sgd]
            L1_sgd= [L1w1, L1w2, L1w3, L1ges_sgd]
            # EVALUATE
            [train_loss_sgd, train_accuracy_sgd]= model.evaluate(x_train, y_train)
            [test_loss_sgd, test_accuracy_sgd] = model.evaluate(x_test, y_test)


    ## Plot ACCURACY
    plt.plot(accuracy_values_multi, 'b')
    plt.plot(accuracy_values_multiadam, 'm')
    plt.plot(accuracy_values_sgd, 'c')
    plt.plot(accuracy_values_adam, 'g')
    xs = np.linspace(1, 21, epochs)
    plt.hlines(y=0.989, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)
    plt.plot(history_multi.history['val_accuracy'], 'b', linestyle= 'dotted')
    plt.plot(history_multiadam.history['val_accuracy'], 'm', linestyle= 'dotted')
    plt.plot(history_sgd.history['val_accuracy'], 'c', linestyle= 'dotted')
    plt.plot(history_adam.history['val_accuracy'], 'g', linestyle= 'dotted')
    plt.title(f'Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train SMGD', 'Train MAdam','Train SGD', 'Train Adam', 'Validate SMGD', 'Validate MAdam', 'Validate SGD', 'Validate Adam'], loc='lower right')
    plt.savefig(f'Acc_mnist-LR-{learning_rate:.4}-{pruning}.png')
    plt.close()

    ## Plot Loss/Loss1
    plt.plot(loss1_values_multi, 'b')
    plt.plot(loss1_values_multiadam, 'm')
    plt.plot(loss_values_sgd, 'c')
    plt.plot(loss_values_adam, 'g')
    plt.plot(history_multi.history['val_loss1'], 'b',linestyle= 'dotted')
    plt.plot(history_multiadam.history['val_loss1'], 'm',linestyle= 'dotted')
    plt.plot(history_sgd.history['val_loss'], 'c',linestyle= 'dotted')
    plt.plot(history_adam.history['val_loss'], 'g',linestyle= 'dotted')
    plt.title(f'Model Loss ')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train SMGD', 'Train MAdam','Train SGD', 'Train Adam', 'Validate SMGD', 'Validate MAdam', 'Validate SGD', 'Validate Adam'], loc='lower right')
    plt.savefig(f'Loss_mnist-LR-{learning_rate:.4}-{pruning}.png')
    plt.close()

    plt.plot(history_multi.history['c2'])
    plt.plot(history_multiadam.history['c2'])
    xs = np.linspace(1, 21, epochs)
    plt.hlines(y=3e-4, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)   
    plt.title('Loss2 Weights Chosen')
    plt.legend(['SMGD', 'MAdam'])
    plt.ylabel('Weighting')
    plt.xlabel('Epoch')
    plt.savefig(f'C2_mnist-LR-{learning_rate:.4}-multi-{pruning}.png')
    plt.close()

    if pruning: 
        ## Plot L0 Values 
        plt.plot(nonzero_weights1_multi, 'b')
        plt.plot(nonzero_weights1_multiadam, 'm')
        plt.plot(nonzero_weights1_sgd, 'c')
        plt.plot(nonzero_weights1_adam, 'g')
        xs = np.linspace(1, 21, 35)
        plt.hlines(y=2107, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)
        plt.title(f'Nonzero Weights Layer1')
        plt.ylabel('Amount of Nonzeros Weights')
        plt.xlabel('Epoch')
        plt.legend(['SMGD', 'MAdam','SGD', 'Adam'], loc='upper right')
        plt.savefig(f'Nonzeros1_mnist-LR-{learning_rate:.4}-pruning.png')
        plt.close()

        plt.plot(nonzero_weights2_multi, 'b')
        plt.plot(nonzero_weights2_multiadam, 'm')
        plt.plot(nonzero_weights2_sgd, 'c')
        plt.plot(nonzero_weights2_adam, 'g')
        xs = np.linspace(1, 21, 35)
        plt.hlines(y=300, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)
        plt.axis([0, 35, 0, 11000])
        plt.title(f'Nonzero Weights Layer2')
        plt.ylabel('Amount of Nonzeros Weights')
        plt.xlabel('Epoch')
        plt.legend(['SMGD', 'MAdam','SGD', 'Adam'], loc='upper right')
        plt.savefig(f'Nonzeros2_mnist-LR-{learning_rate:.4}-pruning.png')
        plt.close()

        plt.plot(nonzero_weights3_multi, 'b')
        plt.plot(nonzero_weights3_multiadam, 'm')
        plt.plot(nonzero_weights3_sgd, 'c')
        plt.plot(nonzero_weights3_adam, 'g')
        xs = np.linspace(1, 21, 35)
        plt.hlines(y=170, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)
        plt.title(f'Nonzero Weights Layer3')
        plt.ylabel('Amount of Nonzeros Weights')
        plt.xlabel('Epoch')
        plt.legend(['SMGD', 'MAdam','SGD', 'Adam'], loc='upper right')
        plt.savefig(f'Nonzeros3_mnist-LR-{learning_rate:.4}-pruning.png')
        plt.close()

    del model
###################################### CIFAR Training ###########################################################
elif cifar10: # test performance of MRMSProp and RMSProp optimizer on the same problem (optional SMGD and SGD as well)
    for opt in optimizers:
        # start with the same init weights
        seed(1)
        set_random_seed(2)
        #collect values while training (in evaluation mode)
        loss1_values = []
        loss2_values = []
        loss_values=[]
        val_loss_values = []
        accuracy_values = []
        val_accuracy_values = []
        nonzero_weights = []

        if opt == 'multi':
            weight_decay=1e-6
            decay_rate = learning_rate / epochs
            momentum = 0.9
            model=vggnetmultimodel(input_shape=input_shape, weight_decay=weight_decay)
            model.mcompile(optimizer=SMGD(multi=True, split=False, learning_rate=learning_rate, descent_weight1=1, descent_weight2=3e-2, momentum=momentum, decay=decay_rate), loss1='sparse_categorical_crossentropy', loss2=L1lossDense(model),metrics=['accuracy'])
        elif opt == 'multirms':
            weight_decay=1e-6
            learning_rate_rmsprop = learning_rate * 0.1
            model=vggnetmultimodel(input_shape=input_shape, weight_decay=weight_decay)
            model.mcompile(optimizer=MRMSprop(multi=True, split=False, learning_rate=learning_rate_rmsprop, descent_weight1=1, descent_weight2=3e-2), loss1='sparse_categorical_crossentropy', loss2=L1lossDense(model),metrics=['accuracy'])
       
        elif opt == 'sgd':
            weight_decay=1e-6
            decay_rate = learning_rate / epochs
            momentum = 0.9
            lambdas=0.03
            model=vggnetregmodel(input_shape= input_shape, weight_decay=weight_decay,lambdas=lambdas)
            model.compile(optimizer=SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        elif opt == 'rms':
            weight_decay=1e-6
            lambdas=0.03
            learning_rate_rmsprop=learning_rate*0.1
            model=vggnetregmodel(input_shape= input_shape, weight_decay=weight_decay,lambdas=lambdas)
            model.compile(optimizer=RMSprop(learning_rate=learning_rate_rmsprop),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        ## Define Callbacks (all thresholds same)
        threshold=0.001
        safe_nonzeroweights=LambdaCallback(on_epoch_end= lambda epoch,
                                    logs: [nonzero_weights.append([np.count_nonzero(
                                                model.get_layer('denselayer').get_weights()[0])])
                                                ]

                                        )

        weight_callback_batch=LambdaCallback(on_batch_end= lambda batch,
                                    logs: [model.get_layer(f"{name}").set_weights(update_weights(
                                                    model.get_layer(f"{name}").get_weights(),threshold))
                                                    for name in ['denselayer']],
                                    on_epoch_end= lambda epoch,
                                    logs: [print(np.count_nonzero(model.get_layer(f"{name}").get_weights()[0]))
                                            for name in ['denselayer']]
                                    )
        if pruning: 
            ## Get, Update and Set Weights before Training
            weights = model.get_layer('denselayer').get_weights()

            sparsified_weights = update_weights(weights, threshold)

            model.get_layer('denselayer').set_weights(sparsified_weights)

            nonzero_weights.append([np.count_nonzero(
                                        model.get_layer('denselayer').get_weights()[0])])


        if opt == 'multi':
            if pruning: 
                history_multi=model.mfit(x_train, y_train, epochs=epochs,batch_size=64, validation_data=[x_test,y_test], callbacks=[weight_callback_batch,safe_nonzeroweights,TestCallback_multi((x_train,y_train))])
                nonzero_weights_multi= nonzero_weights
                nonzero_weights=[]
            else:
                history_multi=model.mfit(x_train, y_train, epochs=epochs,batch_size=64, validation_data=[x_test,y_test], callbacks=[TestCallback_multi((x_train,y_train))])
            accuracy_values_multi= accuracy_values
            accuracy_values=[]
            loss2_values_multi = loss2_values
            loss2_values = []
            loss1_values_multi=loss1_values
            loss1_values=[]
            weights = model.get_layer("denselayer").get_weights() #weights and biases of the layer
            L1w1=sum(sum(sum(np.abs(weights))))
            L0w1=np.count_nonzero(weights[0]) + np.count_nonzero(weights[1])

            L0ges_multi=L0w1
            L1ges_multi= L1w1
            L0L1_multi=[L0ges_multi,L1ges_multi]
            
            # EVALUATE
            [train_loss1_multi, train_loss2_multi, train_accuracy_multi] = model.evaluate_multi(x_train, y_train)
            [test_loss1_multi, test_loss2_multi, test_accuracy_multi] = model.evaluate_multi(x_test, y_test)
        
        elif opt == 'multirms':
            if pruning: 
                history_multirms= model.mfit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_cifar),weight_callback_batch,safe_nonzeroweights,TestCallback_multi((x_train,y_train))])
                nonzero_weights_multirms= nonzero_weights
                nonzero_weights=[]
            else: 
                history_multirms=model.mfit(x_train, y_train, epochs=epochs,batch_size=64, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_cifar),TestCallback_multi((x_train,y_train))])
            
            accuracy_values_multirms= accuracy_values
            accuracy_values=[]
            loss2_values_multirms = loss2_values
            loss2_values = []
            loss1_values_multirms=loss1_values
            loss1_values=[]
            weights = model.get_layer("denselayer").get_weights() 
            L1w1=sum(sum(sum(np.abs(weights))))
            L0w1=np.count_nonzero(weights[0]) + np.count_nonzero(weights[1])

            L0ges_multirms=L0w1
            L1ges_multirms= L1w1
            L0L1_multirms=[L0ges_multirms,L1ges_multirms]
            
            # EVALUATE
            [train_loss1_multirms, train_loss2_multirms, train_accuracy_multirms] = model.evaluate_multi(x_train, y_train)
            [test_loss1_multirms, test_loss2_multirms, test_accuracy_multirms] = model.evaluate_multi(x_test, y_test)
    
        elif opt == 'sgd':
            if pruning: 
                history= model.fit(x_train, y_train,batch_size=64,epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_cifar),weight_callback_batch,safe_nonzeroweights,TestCallbackSGD((x_train,y_train))])
                nonzero_weights_SGD= nonzero_weights
                nonzero_weights=[]
            else: 
                history= model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_cifar),TestCallback((x_train,y_train))])
            accuracy_values_SGD= accuracy_values
            accuracy_values=[]
            loss_values_SGD= loss_values
            loss_values=[]
            weights = model.get_layer("denselayer").get_weights()
            L1w1=sum(sum(sum(np.abs(weights))))
            L0w1=np.count_nonzero(weights[0]) + np.count_nonzero(weights[1])

            L0ges= L0w1
            L1ges= L1w1
            L0L1=[L0ges,L1ges]
            # EVALUATE
            [train_loss, train_accuracy]= model.evaluate(x_train, y_train)
            [test_loss, test_accuracy] = model.evaluate(x_test, y_test)


        elif opt == 'rms':
            if pruning: 
                history_rms= model.fit(x_train, y_train,batch_size=64,epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_cifar),weight_callback_batch,safe_nonzeroweights,TestCallback((x_train,y_train))])
                nonzero_weights_rms= nonzero_weights
                nonzero_weights=[]
            else: 
                history_rms= model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test,y_test], callbacks=[LearningRateScheduler(lr_schedule_cifar),TestCallback((x_train,y_train))])
            accuracy_values_rms= accuracy_values
            accuracy_values=[]
            loss_values_rms= loss_values
            loss_values=[]
            weights = model.get_layer("denselayer").get_weights()
            L1w1=sum(sum(sum(np.abs(weights))))
            L0w1=np.count_nonzero(weights[0]) + np.count_nonzero(weights[1])

            L0ges_rms= L0w1
            L1ges_rms= L1w1
            L0L1_rms=[L0ges_rms,L1ges_rms]
            # EVALUATE
            [train_loss_rms, train_accuracy_rms]= model.evaluate(x_train, y_train)
            [test_loss_rms, test_accuracy_rms] = model.evaluate(x_test, y_test)

           
    ## Plot Accuracy
    plt.plot(accuracy_values_multi, 'g')
    plt.plot(accuracy_values_multirms, 'm')
    plt.plot(accuracy_values_SGD, 'b') 
    plt.plot(accuracy_values_rms, 'c')
    plt.plot(history_multi.history['val_accuracy'], 'g', linestyle= 'dotted')
    plt.plot(history_multirms.history['val_accuracy'], 'm', linestyle= 'dotted')
    plt.plot(history.history['val_accuracy'], color='b', linestyle= 'dotted')
    plt.plot(history_rms.history['val_accuracy'], color='c', linestyle= 'dotted')
    plt.title(f'Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train SMGD','Train MRMSProp','Train SGD','Train RMSProp','Validate SMGD','Validate MRMSProp','Validate SGD','Validate RMSProp'], loc='lower right')
    plt.savefig(f'Acc_multi+pruning_cifar-LR-{learning_rate:.4}-{pruning}.png')
    plt.close()


    ## Plot Loss/Loss1
    plt.plot(loss1_values_multi, 'g')
    plt.plot(loss1_values_multirms, 'm')
    plt.plot(loss_values_SGD,color='b')
    plt.plot(loss_values_rms, color='c')
    plt.plot(history_multi.history['val_loss1'], 'g', linestyle='dotted')
    plt.plot(history_multirms.history['val_loss1'], 'm', linestyle= 'dotted')
    plt.plot(history.history['val_loss'], color= 'b', linestyle='dotted')
    plt.plot(history_rms.history['val_loss'], linestyle= 'dotted',color='c')
    plt.title(f'Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train SMGD','Train MRMSProp','Train SGD','Train RMSProp','Validate SMGD','Validate MRMSProp','Validate SGD','Validate RMSProp'], loc='lower right')
    plt.savefig(f'Loss_cifar-LR-{learning_rate:.4}-{pruning}.png')
    plt.close()


    if pruning: 
        ## Plot L0 Values
        plt.plot(nonzero_weights_multi, 'g')
        plt.plot(nonzero_weights_multirms, 'm')
        plt.plot(nonzero_weights_SGD, 'b')
        plt.plot(nonzero_weights_rms, color='c')
        plt.title(f'Nonzero Weights Layer1 t={learning_rate:.4}')
        plt.ylabel('Amount of Nonzeros Weights')
        plt.xlabel('Epoch')
        plt.legend(['SMGD', 'MRMSProp', 'SGD', 'RMSProp'], loc='upper right')
        plt.savefig(f'Nonzeros_cifar-LR-{learning_rate:.4}-{pruning}.png')
        plt.close()



    plt.plot(history_multi.history['c2'], 'g')
    plt.plot(history_multirms.history['c2'], 'm')
    plt.legend(['SMGD','MRMSProp'])
    plt.title('Loss Weight C2 Chosen')
    plt.ylabel('Weighting')
    plt.xlabel('Epoch')
    plt.savefig(f'C2_cifar-LR-{learning_rate:.4}-multi-{pruning}.png')
    plt.close()


    del model

else:
    print("Please specify a data set!")