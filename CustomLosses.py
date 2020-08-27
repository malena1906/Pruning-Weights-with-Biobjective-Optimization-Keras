'''
 Author      : Malena Reiners, M. Sc. Mathematics
 Description : Definition of Regularization Loss Functions (l1 and l2).
               Toolbox for the application of the stochastic multi-gradient descent algorithm (SMGD version 0.1) combined with pruning neural
               networks weights. We need a l1 or l2 regularization function wrapped as a loss function. Therefore, we cheat a little bit
               as Keras loss functions need the dependency of the y_true's and y_pred'.

 Input(s)    : The model such that one can access the weight vectors.

 Output(s)   : The l1 or l2 value in each iteration, in some cases only of convolution weights or dense layers weights, in others of all weights.
               For the usage of the second loss function in SMGD algorithm.

 Notes       : As we want to use loss functions which are independent of the training data but replace a true loss function, we needed this 
               quite pointless construction of the loss functions. Please make your changes to use whatever loss you whish to minimize. 
               The code is implemented using Python3.7, Keras 2.3.1 and Tensorflow 1.14.0


'''

# Keras libraries
import keras 
import keras.backend as K
import numpy as np 

"""Own Loss Function (L1 loss)""" # problem as usually only dense weights for L1 
def L1loss(model):
    variables=model.trainable_weights
    flattenedList = [K.flatten(x) for x in variables]
    weights = K.concatenate(flattenedList)
    def old_loss(y_true,y_pred): # needed to not also modify "Loss Function Class" of Tensorflow and Keras
        return K.sum(K.abs(weights))
    return old_loss

"""Own Loss Function (L2 loss)"""  # problem as usually only conv weights for L2 
def L2loss(model):
    variables=model.trainable_weights
    flattenedList = [K.flatten(x) for x in variables]
    weights = K.concatenate(flattenedList)
    def old_loss(y_true,y_pred): 
        return K.sum(K.square(weights))
    return old_loss


"""Own Loss Function Dense/Conv (L1L2 loss)""" # problem as the same lambda value (formerly regularization coefficient) for both L1 and L2 
def L1L2lossDenseConv(model):
    dense_weights=[]
    rest_weights=[]
    variables=model.trainable_weights
    for var in variables:
        if 'dense' in var.name:
            dense_weights.append(K.flatten(var))
        else:
            rest_weights.append(K.flatten(var))
    dense_all = K.concatenate(dense_weights)
    rest_all= K.concatenate(rest_weights)
    def old_loss(y_true,y_pred): 
        return K.sum(K.abs(dense_all)) + K.sum(K.square(rest_all))
    return old_loss

"""Own Loss Function (L1 loss) for Dense Weights Only"""
def L1lossDense(model):
    dense_weights=[]
    rest_weights=[]
    variables=model.trainable_weights
    for var in variables:
        if 'dense' in var.name and 'bias' not in var.name:
            dense_weights.append(K.flatten(var))
        else:
            rest_weights.append(K.flatten(var))
    dense_all = K.concatenate(dense_weights)
    rest_all= K.concatenate(rest_weights)
    def old_loss(y_true,y_pred): 
        return K.sum(K.abs(dense_all)) + 0.0 * K.sum(0.0 * K.square(rest_all))
    return old_loss

"""Own Loss Function (L2 loss) for Conv Weights Only"""
def L2lossConv(model):
    dense_weights=[]
    rest_weights=[]
    variables=model.trainable_weights
    for var in variables:
        if 'dense' in var.name:
            dense_weights.append(K.flatten(var))
        else:
            rest_weights.append(K.flatten(var))
    dense_all = K.concatenate(dense_weights)
    rest_all= K.concatenate(rest_weights)
    def old_loss(y_true,y_pred): 
        return 0.0*K.sum(K.abs(dense_all)) + K.sum(K.square(rest_all))
    return old_loss



