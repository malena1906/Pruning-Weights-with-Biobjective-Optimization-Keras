'''
 Author      : Malena Reiners, M. Sc. Mathematics
 Description : LeNet-5 implementation using Keras Libraries adapted from

               LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., 1998. Gradient-based learning applied to document recognition.
               Proceedings of the IEEE 86 (11),2278–2324.

               VGG-like architecture implementation using Keras Libraries adapted from

               Simonyan, K., Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition.
               Published as a conference paper at ICLR 2015

               each with and without l1 regularization in the dense layers and compatible for the usage of stochastic multi gradient
               descent optimizers (SMGD, MAdam, MRMSProp version 0.1). Therefore it is neccessary to import MultiobjectiveOptimizers.py 
               and MultiobjectiveClasses.py

               detailed description of the both network architectures can be found also in the following paper:
               
               Reiners, M., Klamroth, K., Stiglmayr, M., 2020, Efficient and Sparse Neural Networks by Pruning 
               Weights in a Multiobjective Learning Approach, ARXIV LINK HERE

 Input(s)    : The input shape of the data used for training and optional a weight_decay value for l2 regularizers in convolutional layers.
 Output(s)   : Neural network architectures.
 Notes       : The code is implemented using Python 3.7, Keras 2.3.1 and Tensorflow 1.14.0
               Please not that it is mandatory to use these versions of Tensorflow and Keras, otherwise the program cannot be executed.
               The reason for this are the changed and adapted Keras and Tensorflow functions of the SMGD algorithm.

'''
# Keras libraries
import keras
import tensorflow as tf
from keras.engine.training import Model
import numpy as np

# Custom scripts 
import MultiobjectiveOptimizers
from MultiobjectiveClasses import Multi

### for pruning weights in neural networks 
def update_weights(weight_matrix, threshold):
    sparsified_weights = []
    for w in weight_matrix:
        bool_mask = (abs(w) > threshold).astype(int)
        sparsified_weights.append(w * bool_mask)
    return sparsified_weights

### for loading the data of MNIST or CIFAR10; which belongs to the network architectures given below
def get_data(mnist=True, cifar10=False):
    if mnist:
        ## Specify the training duration
        epochs = 30
        ## Load MNIST
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        train_data = (x_train, y_train)
        ## Reshape the array to 4-dims so that it can work with the Keras API
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28,28,1)

        ## Make sure that the values are float so that we can get decimal points after
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')##

        ## Normalize the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        x_test /= 255
        num_classes=10
    elif cifar10:
        ## Specify the training duration
        epochs = 125
        ## Load CIFAR
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        train_data = (x_train, y_train)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        input_shape = x_train.shape[1:]
        # z-score
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)
        num_classes = 10

    return x_train, y_train, x_test, y_test, train_data, input_shape, epochs, num_classes

### LeNet-5 architecutre for a multi objective optimization Model to train on MNIST
def lenet5multimodel(input_shape, weight_decay):
    inputs1 = keras.layers.Input(shape=input_shape)
    output1= keras.layers.Conv2D(6, kernel_size=(3,3),input_shape=input_shape,kernel_regularizer=keras.regularizers.l2(weight_decay), activation='relu')(inputs1)
    output2= keras.layers.AveragePooling2D(pool_size=(2, 2))(output1)
    output3= keras.layers.Conv2D(16,kernel_size=(3, 3),activation='relu',kernel_regularizer=keras.regularizers.l2(weight_decay))(output2)
    output4= keras.layers.AveragePooling2D(pool_size=(2, 2))(output3)
    output5= keras.layers.Flatten()(output4)
    output6= keras.layers.Dense(120, activation='relu', name='denselayer1')(output5)
    output7= keras.layers.Dense(84, activation='relu', name='denselayer2')(output6)
    predictions= keras.layers.Dense(10, activation='softmax', name='denselayer3')(output7)

    return Multi(inputs=inputs1, outputs=predictions)

### LeNet-5 architecutre for a single objective optimization Model to train on MNIST
def lenet5regmodel(input_shape, weight_decay, lambdas):
    inputs1 = keras.layers.Input(shape=input_shape)
    output1= keras.layers.Conv2D(6, kernel_size=(3,3),input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(weight_decay), activation='relu')(inputs1)
    output2= keras.layers.AveragePooling2D(pool_size=(2, 2))(output1)
    output3= keras.layers.Conv2D(16,kernel_size=(3, 3),activation='relu',kernel_regularizer=keras.regularizers.l2(weight_decay))(output2)
    output4= keras.layers.AveragePooling2D(pool_size=(2, 2))(output3)
    output5= keras.layers.Flatten()(output4)
    output6= keras.layers.Dense(120, activation='relu',kernel_regularizer=keras.regularizers.l1(lambdas), name='denselayer1')(output5)
    output7= keras.layers.Dense(84, activation='relu', kernel_regularizer=keras.regularizers.l1(lambdas), name='denselayer2')(output6)
    predictions= keras.layers.Dense(10, activation='softmax', kernel_regularizer=keras.regularizers.l1(lambdas), name='denselayer3')(output7)

    return Model(inputs=inputs1, outputs=predictions)

### VGG-like architecutre for a multi objective optimization Model to train on CIFAR10
def vggnetmultimodel(input_shape, weight_decay):
    inputs1 = keras.layers.Input(shape=input_shape)
    output1 = keras.layers.Conv2D(32, kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(weight_decay), padding='same', input_shape=input_shape, activation="elu")(inputs1)
    output2 = keras.layers.BatchNormalization()(output1)
    output3 = keras.layers.Conv2D(32, kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(weight_decay), padding='same',activation="elu")(output2)
    output4 = keras.layers.BatchNormalization()(output3)
    output5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(output4)
    output6 = keras.layers.Dropout(0.2)(output5)
    output7 = keras.layers.Conv2D(64, (3, 3), kernel_regularizer=keras.regularizers.l2(weight_decay), padding='same', activation='elu')(output6)
    output8 = keras.layers.BatchNormalization()(output7)
    output9 = keras.layers.Conv2D(64, (3, 3), kernel_regularizer=keras.regularizers.l2(weight_decay), padding='same',activation='elu')(output8)
    output10 = keras.layers.BatchNormalization()(output9)
    output11 = keras.layers.MaxPooling2D(pool_size=(2, 2))(output10)
    output12 = keras.layers.Dropout(0.3)(output11)
    output13 = keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), activation="elu")(output12)
    output14 = keras.layers.BatchNormalization()(output13)
    output15 = keras.layers.Conv2D(128, kernel_size=(3, 3),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), activation="elu")(output14)
    output16 = keras.layers.BatchNormalization()(output15)
    output17 = keras.layers.MaxPooling2D(pool_size=(2, 2))(output16)
    output18 = keras.layers.Dropout(0.4)(output17)
    output19 = keras.layers.Flatten()(output18)
    predictions = keras.layers.Dense(10, activation='softmax', name='denselayer')(output19)

    return Multi(inputs=inputs1, outputs=predictions)

### VGG-like architecutre for a single objective optimization Model to train on CIFAR10
def vggnetregmodel(input_shape, weight_decay, lambdas):
    inputs1 = keras.layers.Input(shape=input_shape)
    output1 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay), input_shape=input_shape, activation="elu")(inputs1)
    output2 = keras.layers.BatchNormalization()(output1)
    output3 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay),activation="elu")(output2)
    output4 = keras.layers.BatchNormalization()(output3)
    output5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(output4)
    output6 = keras.layers.Dropout(0.2)(output5)
    output7 = keras.layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay), activation='elu')(output6)
    output8 = keras.layers.BatchNormalization()(output7)
    output9 = keras.layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay),activation='elu')(output8)
    output10 = keras.layers.BatchNormalization()(output9)
    output11 = keras.layers.MaxPooling2D(pool_size=(2, 2))(output10)
    output12 = keras.layers.Dropout(0.3)(output11)
    output13 = keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay), activation="elu")(output12)
    output14 = keras.layers.BatchNormalization()(output13)
    output15 = keras.layers.Conv2D(128, kernel_size=(3, 3),padding='same',kernel_regularizer=keras.regularizers.l2(weight_decay), activation="elu")(output14)
    output16 = keras.layers.BatchNormalization()(output15)
    output17 = keras.layers.MaxPooling2D(pool_size=(2, 2))(output16)
    output18 = keras.layers.Dropout(0.4)(output17)
    output19 = keras.layers.Flatten()(output18)
    predictions = keras.layers.Dense(10, activation='softmax',kernel_regularizer=keras.regularizers.l1(lambdas), name='denselayer')(output19)

    return Model(inputs=inputs1, outputs=predictions)
        
