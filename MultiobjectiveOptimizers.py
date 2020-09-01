'''
 Author      :  Malena Reiners, M. Sc. Mathematics
 Description :  Stochastic multi-gradient descent algorithm (SMGD version 0.1) implementation in Keras and Tensorflow  
                Applied not only for vanilla SGD, but also for Adam Optimizer (MAdam version 0.1) and RMSprop optimizer (MRMSprop version 0.1).

                To use these optimizers the MultiobjectiveClasses.py is needed to process two objectives and so on. 

                Refer to these papers for more details about the multi-gradient algorithm: 

                S. Liu and L. N. Vicente, The stochastic multi-gradient algorithm for multi-objective optimization and its 
                application to supervised machine learning, ISE Technical Report 19T-011, Lehigh University.

                and for the success using different extensions and combine the MOP training with pruning weights:

                Reiners, M., Klamroth, K., Stiglmayr, M., 2020, Efficient and Sparse Neural Networks by Pruning 
                Weights in a Multiobjective Learning Approach, https://arxiv.org/abs/2008.13590


 Input(s)    :  A neural network architecture implemented as a Keras model with two loss functions.
 Output(s)   :  A Pareto stationary point for the trade-off between both objective functions (loss functions)
 Notes       :  The code is implemented using Python 3.7, Keras 2.3.1 and Tensorflow 1.14.0

                Please not that it is mandatory to use these versions of Tensorflow and Keras, otherwise the program cannot be executed. 
                The reasons for this are the changed and adapted Keras and Tensorflow functions of this particular versions. 

                To extend it to multiple objectives (more than two), one may need additional packages, e.g., Gurobi, to solve a quadratic
                subproblem and compute a common descent direction of multi-objective at current point.
            
'''
import warnings
from scipy.sparse import issparse
from six.moves import zip
import matplotlib.pyplot as plt
import collections
import keras
from keras import backend as K
from keras import losses, optimizers
from keras import metrics as metrics_module
from keras.engine import  training_arrays,training_utils
from keras.engine.training import Model
from keras.optimizers import Optimizer
from keras.utils import losses_utils
from keras.utils import np_utils
from keras.utils.generic_utils import (slice_arrays, to_list)
from keras.engine.training_utils import batch_shuffle
from keras.engine.training_utils import check_num_samples
from keras.engine.training_utils import make_batches
from keras.engine.training_utils import should_run_validation
from keras import callbacks as cbks
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import unpack_singleton
from keras import regularizers
from keras.callbacks import LearningRateScheduler, LambdaCallback, Callback

### custom scripts 
from CustomLosses import L1loss, L2loss, L1lossDense, L2lossConv, L1L2lossDenseConv
import MultiobjectiveClasses

import numpy as np
if K.backend() == 'tensorflow':
    import tensorflow as tf

# print(tf.__version__) #1.14.0
# print(keras.__version__) #2.3.1


"""SMGD Optimizer written for Keras"""
class SMGD(Optimizer):
    """Stochastic multi gradient descent optimizer.

    Implemented only for bicriteria problems so far, 
    with the possibility to divide the second descent weight parameter into 2 components, e.g.
    for the usage of different strong regularizations on dense or conv layer. 
    loss3 per default L1 and loss4 per default L2 regularization of all weights, 
    loss5 per default L1Dense and loss6 per default L2Dense to split regularizartion. 
    For more information on the used loss functions, see CustomLosses.py

    Includes support for momentum,
    learning rate decay and Nesterov momentum

    # Arguments
        learning_rate: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        descent_weight1: >=0 und <= 1. Weight to calculate common descent direction from loss1.
        descent_weight2: >=0 und <= 1. Weight to calculate common descent direction from loss2.
        nesterov: boolean. Whether to apply Nesterov momentum.
        multi: boolean. Whether we want to have the weighthing be calculated. (If False: provide descent_weights).
        split: boolean. Whether we want to distinguish between which loss is used in different layers (conv/dense).

     # References
        - Multigradient Optimizer, implemented from the paper: [The stochastic multi-gradient algorithm for 
        multi-objective optimization and its application to supervised machine learning]
        (http://www.optimization-online.org/DB_FILE/2019/07/7282.pdf)
    """

    def __init__(self, learning_rate=0.01, momentum=0., descent_weight1=0.5, descent_weight2=0.5,
                 nesterov=False, multi=False, split=False, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        self.initial_decay = kwargs.pop('decay', 0.01) 
        super(SMGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.descent_weight1 = K.variable(descent_weight1, name='descent_weight1')
            self.descent_weight2 = K.variable(descent_weight2, name='descent_weight2')
        self.nesterov = nesterov
        self.multi = multi
        self.split= split #split weights into dense/conv weights

    def get_updates(self, loss1, loss2, loss3, loss4, loss5, loss6, params):
        grads1 = self.get_gradients(loss1, params)
        grads2 = self.get_gradients(loss2, params)
        grads5 = self.get_gradients(loss5, params) # l1 loss dense 
        grads6= self.get_gradients(loss6, params) # l2 loss conv 

        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.learning_rate

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        c1 = self.descent_weight1
        c2 = self.descent_weight2
        ## for split and without multi specify the splitted weighting 
        c11 = c1 # for CE dense 
        c21 = c2 # for l1 dense 
        c12 = 1  # for CE conv 
        c22 = 4e-1 # for l2 dense 


        if self.multi and not self.split: # calculate weighting for the loss functions given (default, also in the paper)
            zero = K.variable(0, name='zero')
            one = K.variable(1, name='one')

            flattenedList1 = [K.flatten(x) for x in grads1]
            gradients1 = K.concatenate(flattenedList1)
            flattenedList2 = [K.flatten(x) for x in grads2]
            gradients2 = K.concatenate(flattenedList2)

            grad21 = gradients2 - gradients1
            grad12 = gradients1 - gradients2
            z1 = K.sum(grad21 * gradients2)
            z2 = K.sum(grad12 * gradients1)
            n = K.sum(grad21 * grad21)

            cm1 = z1 / n
            c1 = K.switch(K.equal(K.all(K.equal(gradients1, gradients2)), K.constant(True, dtype=bool)),
                          lambda: one, lambda: cm1)
            cm2 = z2 / n
            c2 = K.switch(K.equal(K.all(K.equal(gradients1, gradients2)), K.constant(True, dtype =bool)),lambda: zero, lambda: cm2)
           
            (c1, c2) = K.switch(c1 < 0, lambda: (zero, one), lambda: (c1, c2))
            (c2, c1) = K.switch(c2 < 0, lambda: (zero, one), lambda: (c2, c1))

        if self.split and self.multi:  # calculate weighting for the loss1 given but split in conv/dense and use different loss2 (namely split loss 2 in loss5 and loss6)
            zero = K.variable(0, name='zero')
            one = K.variable(1, name='one')

            flattenedList1 = [K.flatten(x) for x in grads1]
            gradients1 = K.concatenate(flattenedList1)
            flattenedList5 = [K.flatten(x) for x in grads5]
            gradients5 = K.concatenate(flattenedList5)
            flattenedList6 = [K.flatten(x) for x in grads6]
            gradients6 = K.concatenate(flattenedList6)

            grad51 = gradients5 - gradients1
            grad15 = gradients1 - gradients5
            z1 = K.sum(grad51 * gradients5)
            z2 = K.sum(grad15 * gradients1)
            n = K.sum(grad51 * grad51)

            cm1 = z1 / n
            c11 = K.switch(K.equal(K.all(K.equal(gradients1, gradients5)), K.constant(True, dtype=bool)),
                          lambda: one, lambda: cm1)
            cm2 = z2 / n
            c21 = K.switch(K.equal(K.all(K.equal(gradients1, gradients5)), K.constant(True, dtype =bool)),lambda: zero, lambda: cm2)

            (c11, c21) = K.switch(c11 < 0, lambda: (zero, one), lambda: (c11, c21))
            (c21, c11) = K.switch(c21 < 0, lambda: (zero, one), lambda: (c21, c11))

            grad61 = gradients6 - gradients1
            grad16 = gradients1 - gradients6
            z1 = K.sum(grad61 * gradients6)
            z2 = K.sum(grad16 * gradients1)
            n = K.sum(grad61 * grad61)

            cm1 = z1 / n
            c12 = K.switch(K.equal(K.all(K.equal(gradients1, gradients6)), K.constant(True, dtype=bool)),
                            lambda: one, lambda: cm1) # for CE conv
            cm2 = z2 / n 
            c22 = K.switch(K.equal(K.all(K.equal(gradients1, gradients6)), K.constant(True, dtype =bool)),
                            lambda: zero, lambda: cm2) # for l2 conv

            (c12, c22) = K.switch(c12 < 0, lambda: (zero, one), lambda: (c12, c22))
            (c22, c12) = K.switch(c22 < 0, lambda: (zero, one), lambda: (c22, c12))

            c1= c11 # for CE dense 
            c2= c21 # for l1 dense 

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape, name='moment_' + str(i))
                   for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + moments
        if not self.split: 
            for p, g1, g2, m in zip(params, grads1, grads2, moments):

                v = self.momentum * m - lr*(c1*g1+c2*g2) # velocity
                self.updates.append(K.update(m, v))

                if self.nesterov:
                    new_p = p + self.momentum * v - lr*(c1*g1+c2*g2)
                else:
                    new_p = p + v

                # Apply constraints.
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)

                self.updates.append(K.update(p, new_p))
        else: 
            for p, g1, g5, g6, m in zip(params, grads1, grads5, grads6, moments):

                if g6 == 0: # its a dense param
                    v = self.momentum * m - lr*(c11*g1+ c21*g5) # velocity
                    self.updates.append(K.update(m, v))

                    if self.nesterov:
                        new_p = p + self.momentum * v - lr*(c11*g1+ c21*g5) 
                    else:
                        new_p = p + v
                else:  # its a conv param
                    v = self.momentum * m - lr*(c12*g1+ c22*g6) # velocity
                    self.updates.append(K.update(m, v))
                    
                    if self.nesterov:
                        new_p = p + self.momentum * v - lr*(c12*g1+ c22*g6) 
                    else:
                        new_p = p + v

                # Apply constraints.
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)

                self.updates.append(K.update(p, new_p))
        self.c1=c1
        self.c2=c2
        return self.updates, c1, c2

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'descent_weight1': float(K.get_value(self.c1)),
                  'descent_weight2': float(K.get_value(self.c2)),
                  'nesterov': self.nesterov}
        base_config = super(SMGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


"""Multi Adam Optimizer written for Keras"""
class MAdam(Optimizer):
    """Multi Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        descent_weight1: >=0 und <= 1. Weight to calculate common descent direction from loss1.
        descent_weight2: >=0 und <= 1. Weight to calculate common descent direction from loss2.
        multi: boolean. Whether we want to have the weighthing be calculated or not. (If False: provide descent_weights).
        split: boolean. Whether we want to distinguish between which loss is used in different layers(conv/dense).

    # References
        - [Adam - A Method for Stochastic Optimization]
        (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
        (https://openreview.net/forum?id=ryQu7f-RZ)
        - [The stochastic multi-gradient algorithm for multi-objective optimization 
        and its application to supervised machine learning]
        (http://www.optimization-online.org/DB_FILE/2019/07/7282.pdf)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, descent_weight1=0.5, descent_weight2=0.5,
                 amsgrad=False, multi=False, split=False,**kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(MAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.descent_weight1 = K.variable(descent_weight1, name='descent_weight1')
            self.descent_weight2 = K.variable(descent_weight2, name='descent_weight2')
        self.amsgrad = amsgrad
        self.multi = multi
        self.split=split


    def get_updates(self, loss1, loss2, loss3, loss4, loss5, loss6, params):
        grads1 = self.get_gradients(loss1, params)
        grads2 = self.get_gradients(loss2, params)

        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        c1 = self.descent_weight1
        c2 = self.descent_weight2
        ## for split and not multi specify the splitted weighting 
        c11 = c1 # for CE dense 
        c21 = c2 # for l1 dense 
        c12 = 1  # for CE conv 
        c22 = 4e-1 # for l2 dense 

        if self.multi and not self.split: # calculate weighting for the loss functions given (should be default)

            zero = K.variable(0, name='zero')
            one = K.variable(1, name='one')

            flattenedList1 = [K.flatten(x) for x in grads1]
            gradients1 = K.concatenate(flattenedList1)
            flattenedList2 = [K.flatten(x) for x in grads2]
            gradients2 = K.concatenate(flattenedList2)

            grad21 = gradients2 - gradients1
            grad12 = gradients1 - gradients2
            z1 = K.sum(grad21 * gradients2)
            z2 = K.sum(grad12 * gradients1)
            n = K.sum(grad21 * grad21)

            cm1 = z1 / n
            c1 = K.switch(K.equal(K.all(K.equal(gradients1, gradients2)), K.constant(True, dtype=bool)),
                          lambda: one, lambda: cm1)
            cm2 = z2 / n
            c2 = K.switch(K.equal(K.all(K.equal(gradients1, gradients2)), K.constant(True, dtype=bool)),
                          lambda: zero, lambda: cm2)
           
            (c1, c2) = K.switch(c1 < 0, lambda: (zero, one), lambda: (c1, c2))
            (c2, c1) = K.switch(c2 < 0, lambda: (zero, one), lambda: (c2, c1))

        if self.split and self.multi: # calculate weighting for the loss1 given but split in conv/dense and use different loss2 (namely split loss 2 in loss5 and loss6)
            zero = K.variable(0, name='zero')
            one = K.variable(1, name='one')

            grads5 = self.get_gradients(loss5, params) # l1 loss dense 
            grads6= self.get_gradients(loss6, params) # l2 loss conv 

            flattenedList1 = [K.flatten(x) for x in grads1]
            gradients1 = K.concatenate(flattenedList1)
            flattenedList5 = [K.flatten(x) for x in grads5]
            gradients5 = K.concatenate(flattenedList5)
            flattenedList6 = [K.flatten(x) for x in grads6]
            gradients6 = K.concatenate(flattenedList6)

            grad51 = gradients5 - gradients1
            grad15 = gradients1 - gradients5
            z1 = K.sum(grad51 * gradients5)
            z2 = K.sum(grad15 * gradients1)
            n = K.sum(grad51 * grad51)

            cm1 = z1 / n
            c11 = K.switch(K.equal(K.all(K.equal(gradients1, gradients5)), K.constant(True, dtype=bool)),
                          lambda: one, lambda: cm1)
            cm2 = z2 / n
            c21 = K.switch(K.equal(K.all(K.equal(gradients1, gradients5)), K.constant(True, dtype =bool)),lambda: zero, lambda: cm2)

            (c11, c21) = K.switch(c11 < 0, lambda: (zero, one), lambda: (c11, c21))
            (c21, c11) = K.switch(c21 < 0, lambda: (zero, one), lambda: (c21, c11))

            grad61 = gradients6 - gradients1
            grad16 = gradients1 - gradients6
            z1 = K.sum(grad61 * gradients6)
            z2 = K.sum(grad16 * gradients1)
            n = K.sum(grad61 * grad61)

            cm1 = z1 / n
            c12 = K.switch(K.equal(K.all(K.equal(gradients1, gradients6)), K.constant(True, dtype=bool)),
                            lambda: one, lambda: cm1) # for CE conv
            cm2 = z2 / n 
            c22 = K.switch(K.equal(K.all(K.equal(gradients1, gradients6)), K.constant(True, dtype =bool)),
                            lambda: zero, lambda: cm2) # for l2 conv

            (c12, c22) = K.switch(c12 < 0, lambda: (zero, one), lambda: (c12, c22))
            (c22, c12) = K.switch(c22 < 0, lambda: (zero, one), lambda: (c22, c12))

            c1= c11 # for CE dense 
            c2= c21 # for l1 dense 

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms1 = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='m_' + str(i))
              for (i, p) in enumerate(params)]
        ms2 = [K.zeros(K.int_shape(p),
                      dtype=K.dtype(p),
                      name='m_' + str(i))
              for (i, p) in enumerate(params)]
        ms6 = [K.zeros(K.int_shape(p),
                      dtype=K.dtype(p),
                      name='m_' + str(i))
              for (i, p) in enumerate(params)]
        vs1 = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='v_' + str(i))
              for (i, p) in enumerate(params)]
        vs2 = [K.zeros(K.int_shape(p),
                      dtype=K.dtype(p),
                      name='v_' + str(i))
              for (i, p) in enumerate(params)]
        vs6 = [K.zeros(K.int_shape(p),
                      dtype=K.dtype(p),
                      name='v_' + str(i))
              for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats1 = [K.zeros(K.int_shape(p),
                     dtype=K.dtype(p),
                     name='vhat_' + str(i))
                     for (i, p) in enumerate(params)]
            vhats2 = [K.zeros(K.int_shape(p),
                            dtype=K.dtype(p),
                            name='vhat_' + str(i))
                    for (i, p) in enumerate(params)]
            vhats6 = [K.zeros(K.int_shape(p),
                            dtype=K.dtype(p),
                            name='vhat_' + str(i))
                    for (i, p) in enumerate(params)]
        else:
            vhats1 = [K.zeros(1, name='vhat_' + str(i))
                     for i in range(len(params))]
            vhats2 = [K.zeros(1, name='vhat_' + str(i))
                      for i in range(len(params))]
            vhats6 = [K.zeros(1, name='vhat_' + str(i))
                      for i in range(len(params))]
        self.weights = [self.iterations] + ms1 + vs1 + vhats1
        if not self.split:  #grads1,2
            for p, g1,g2, m1,v1, vhat1,m2,v2, vhat2 in zip(params, grads1, grads2, ms1, vs1, vhats1, ms2, vs2, vhats2):

                m_t1 = (self.beta_1 * m1) + (1. - self.beta_1) * g1
                m_t2 = (self.beta_1 * m2) + (1. - self.beta_1) * g2
                v_t1 = (self.beta_2 * v1) + (1. - self.beta_2) * K.square(g1)
                v_t2 = (self.beta_2 * v2) + (1. - self.beta_2) * K.square(g2)

                if self.amsgrad:
                    vhat_t1 = K.maximum(vhat1, v_t1)
                    vhat_t2= K.maximum(vhat2, v_t2)
                    p_t = p - lr_t * (c1*(m_t1 / (K.sqrt(vhat_t1) + self.epsilon))+c2*(m_t2 / (K.sqrt(vhat_t2) + self.epsilon)))
                    self.updates.append(K.update(vhat1, vhat_t1))
                    self.updates.append(K.update(vhat2, vhat_t2))
                else:
                    p_t = p - lr_t * (c1*(m_t1 / (K.sqrt(v_t1) + self.epsilon))+c2*(m_t2 / (K.sqrt(v_t2) + self.epsilon)))

                self.updates.append(K.update(m1, m_t1))
                self.updates.append(K.update(m2, m_t2))
                self.updates.append(K.update(v1, v_t1))
                self.updates.append(K.update(v2, v_t2))
                new_p = p_t

                # Apply constraints.
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)

                self.updates.append(K.update(p, new_p))
        else: #grads 1,5,6
             for p, g1, g5, g6, m1,v1, vhat1,m5,v5, vhat5, m6,v6, vhat6 in zip(params, grads1, grads5, grads6, ms1, vs1, vhats1, ms2, vs2, vhats2, ms6, vs6, vhats6):

                m_t1 = (self.beta_1 * m1) + (1. - self.beta_1) * g1
                m_t5 = (self.beta_1 * m5) + (1. - self.beta_1) * g5
                m_t6 = (self.beta_1 * m5) + (1. - self.beta_1) * g6
                v_t1 = (self.beta_2 * v1) + (1. - self.beta_2) * K.square(g1)
                v_t5 = (self.beta_2 * v5) + (1. - self.beta_2) * K.square(g5)
                v_t6 = (self.beta_2 * v6) + (1. - self.beta_2) * K.square(g6)

                if g6 == 0: # its a dense param     
                    if self.amsgrad:
                        vhat_t1 = K.maximum(vhat1, v_t1)
                        vhat_t5= K.maximum(vhat5, v_t5)
                        p_t = p - lr_t * (c11*(m_t1 / (K.sqrt(vhat_t1) + self.epsilon))+c21*(m_t5 / (K.sqrt(vhat_t5)+ self.epsilon)))
                        self.updates.append(K.update(vhat1, vhat_t1))
                        self.updates.append(K.update(vhat5, vhat_t5))
                    else:
                        p_t = p - lr_t * (c11*(m_t1 / (K.sqrt(v_t1) + self.epsilon))+c21*(m_t5 / (K.sqrt(v_t5)+ self.epsilon)))

                    self.updates.append(K.update(m1, m_t1))
                    self.updates.append(K.update(v1, v_t1))
                    self.updates.append(K.update(m5, m_t5))
                    self.updates.append(K.update(v5, v_t5))
                    new_p = p_t
                else:  # its a conv param
                    if self.amsgrad:
                        vhat_t1 = K.maximum(vhat1, v_t1)
                        vhat_t6= K.maximum(vhat6, v_t6)
                        p_t = p - lr_t * (c12*(m_t1 / (K.sqrt(vhat_t1) + self.epsilon))+c22*(m_t6 / (K.sqrt(vhat_t6) + self.epsilon)))
                        self.updates.append(K.update(vhat1, vhat_t1))
                        self.updates.append(K.update(vhat6, vhat_t6))
                    else:
                        p_t = p - lr_t * (c12*(m_t1 / (K.sqrt(v_t1) + self.epsilon))+c22*(m_t6 / (K.sqrt(v_t6) + self.epsilon)))

                    self.updates.append(K.update(m1, m_t1))
                    self.updates.append(K.update(v1, v_t1))
                    self.updates.append(K.update(m6, m_t6))
                    self.updates.append(K.update(v6, v_t6))
                    new_p = p_t
                # Apply constraints.
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)

                self.updates.append(K.update(p, new_p))
        return self.updates,c1,c2

"""Multi RMSprop Optimizer written for Keras"""
class MRMSprop(Optimizer):
    """Multi RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    # Arguments
        learning_rate: float >= 0. Learning rate.
        rho: float >= 0.
        descent_weight1: >=0 und <= 1. Weight to calculate common descent direction from loss1.
        descent_weight2: >=0 und <= 1. Weight to calculate common descent direction from loss2.
        multi: boolean. Whether we want to have the weighthing be calculated or not. (If False: provide descent_weights).
        split: boolean. Whether we want to distinguish between which loss is used in different layers(conv/dense).
    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude
           ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
        - [The stochastic multi-gradient algorithm for multi-objective optimization 
        and its application to supervised machine learning]
        (http://www.optimization-online.org/DB_FILE/2019/07/7282.pdf)
    """

    def __init__(self, learning_rate=0.001, rho=0.9, descent_weight1=0.5, descent_weight2=0.5, multi=False, split=False, **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(MRMSprop, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.descent_weight1 = K.variable(descent_weight1, name="descent_weight1")
            self.descent_weight2 = K.variable(descent_weight2, name="descent_weight2")
        self.split=False
        self.multi= multi

    def get_updates(self, loss1, loss2, loss3, loss4, loss5, loss6, params):
        grads1 = self.get_gradients(loss1, params)
        grads2= self.get_gradients(loss2, params)
        accumulators1 = [K.zeros(K.int_shape(p),
                        dtype=K.dtype(p),
                        name='accumulator_' + str(i))
                        for (i, p) in enumerate(params)]
        accumulators2 = [K.zeros(K.int_shape(p),
                                 dtype=K.dtype(p),
                                 name='accumulator_' + str(i))
                         for (i, p) in enumerate(params)]

        accumulators6 = [K.zeros(K.int_shape(p),
                                 dtype=K.dtype(p),
                                 name='accumulator_' + str(i))
                         for (i, p) in enumerate(params)]

        self.weights = [self.iterations] + accumulators1
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                   K.dtype(self.decay))))
        c1 = self.descent_weight1
        c2 = self.descent_weight2
        ## for split and not multi specify the splitted weighting 
        c11 = c1 # for CE dense 
        c21 = c2 # for l1 dense 
        c12 = 1  # for CE conv 
        c22 = 4e-1 # for l2 dense 

        if self.multi and not self.split: # calculate weighting for the loss functions given (should be default)
            zero = K.variable(0, name='zero')
            one = K.variable(1, name='one')

            flattenedList1 = [K.flatten(x) for x in grads1]
            gradients1 = K.concatenate(flattenedList1)
            flattenedList2 = [K.flatten(x) for x in grads2]
            gradients2 = K.concatenate(flattenedList2)

            grad21 = gradients2 - gradients1
            grad12 = gradients1 - gradients2
            z1 = K.sum(grad21 * gradients2)
            z2 = K.sum(grad12 * gradients1)
            n = K.sum(grad21 * grad21)

            cm1 = z1 / n
            c1 = K.switch(K.equal(K.all(K.equal(gradients1, gradients2)), K.constant(True, dtype=bool)),
                          lambda: one, lambda: cm1)
            cm2 = z2 / n
            c2 = K.switch(K.equal(K.all(K.equal(gradients1, gradients2)), K.constant(True, dtype=bool)),
                          lambda: zero, lambda: cm2)
            (c1, c2) = K.switch(c1 < 0, lambda: (zero, one), lambda: (c1, c2))
            (c2, c1) = K.switch(c2 < 0, lambda: (zero, one), lambda: (c2, c1))

        if self.split and self.multi: # calculate weighting for the loss1 given but split in conv/dense and use different loss2 (namely split loss 2 in loss5 and loss6)
            zero = K.variable(0, name='zero')
            one = K.variable(1, name='one')

            grads5 = self.get_gradients(loss5, params) # l1 loss dense 
            grads6= self.get_gradients(loss6, params) # l2 loss conv 

            flattenedList1 = [K.flatten(x) for x in grads1]
            gradients1 = K.concatenate(flattenedList1)
            flattenedList5 = [K.flatten(x) for x in grads5]
            gradients5 = K.concatenate(flattenedList5)
            flattenedList6 = [K.flatten(x) for x in grads6]
            gradients6 = K.concatenate(flattenedList6)

            grad51 = gradients5 - gradients1
            grad15 = gradients1 - gradients5
            z1 = K.sum(grad51 * gradients5)
            z2 = K.sum(grad15 * gradients1)
            n = K.sum(grad51 * grad51)

            cm1 = z1 / n
            c11 = K.switch(K.equal(K.all(K.equal(gradients1, gradients5)), K.constant(True, dtype=bool)),
                          lambda: one, lambda: cm1)
            cm2 = z2 / n
            c21 = K.switch(K.equal(K.all(K.equal(gradients1, gradients5)), K.constant(True, dtype =bool)),lambda: zero, lambda: cm2)

            (c11, c21) = K.switch(c11 < 0, lambda: (zero, one), lambda: (c11, c21))
            (c21, c11) = K.switch(c21 < 0, lambda: (zero, one), lambda: (c21, c11))

            grad61 = gradients6 - gradients1
            grad16 = gradients1 - gradients6
            z1 = K.sum(grad61 * gradients6)
            z2 = K.sum(grad16 * gradients1)
            n = K.sum(grad61 * grad61)

            cm1 = z1 / n
            c12 = K.switch(K.equal(K.all(K.equal(gradients1, gradients6)), K.constant(True, dtype=bool)),
                            lambda: one, lambda: cm1) # for CE conv
            cm2 = z2 / n 
            c22 = K.switch(K.equal(K.all(K.equal(gradients1, gradients6)), K.constant(True, dtype =bool)),
                            lambda: zero, lambda: cm2) # for l2 conv

            (c12, c22) = K.switch(c12 < 0, lambda: (zero, one), lambda: (c12, c22))
            (c22, c12) = K.switch(c22 < 0, lambda: (zero, one), lambda: (c22, c12))

            c1= c11 # for CE dense 
            c2= c21 # for l1 dense 

        if not self.split:  #grads1,2
            for p, g1, g2, a1,a2 in zip(params, grads1, grads2, accumulators1, accumulators2):
                # update accumulator
                new_a1 = self.rho * a1 + (1. - self.rho) * K.square(g1)
                new_a2 = self.rho * a2 + (1. - self.rho) * K.square(g2)
                self.updates.append(K.update(a1, new_a1))
                self.updates.append(K.update(a2, new_a2))
                new_p = p - lr *( c1*(g1 / (K.sqrt(new_a1) + self.epsilon))+c2*(g2/ (K.sqrt(new_a2) + self.epsilon)))

                # Apply constraints.
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)

                self.updates.append(K.update(p, new_p))
        else: #grads 1,5,6
            for p, g1, g5, g6, a1,a5, a6  in zip(params, grads1, grads5, grads6, accumulators1, accumulators2, accumulators6):
                
                if g6 == 0: # its a dense param     
                    # update accumulator
                    new_a1 = self.rho * a1 + (1. - self.rho) * K.square(g1)
                    new_a5 = self.rho * a5 + (1. - self.rho) * K.square(g5)
                    self.updates.append(K.update(a1, new_a1))
                    self.updates.append(K.update(a5, new_a5))
                    new_p = p - lr *( c11*(g1 / (K.sqrt(new_a1) + self.epsilon))+c21*(g5/ (K.sqrt(new_a5) + self.epsilon)))

                    # Apply constraints.
                    if getattr(p, 'constraint', None) is not None:
                        new_p = p.constraint(new_p)

                    self.updates.append(K.update(p, new_p))
                else: # its a conv param
                    # update accumulator
                    new_a1 = self.rho * a1 + (1. - self.rho) * K.square(g1)
                    new_a6 = self.rho * a6 + (1. - self.rho) * K.square(g6)
                    self.updates.append(K.update(a1, new_a1))
                    self.updates.append(K.update(a6, new_a6))
                    new_p = p - lr *( c12*(g1 / (K.sqrt(new_a1) + self.epsilon))+c22*(g6/ (K.sqrt(new_a6) + self.epsilon)))

                    # Apply constraints.
                    if getattr(p, 'constraint', None) is not None:
                        new_p = p.constraint(new_p)

                    self.updates.append(K.update(p, new_p))

        return self.updates,c1,c2
