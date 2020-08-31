'''
 Author      : Malena Reiners, M.Sc. Mathematics

 Description : Application of the (stochastic) dichotomic search algorithm on the Multi Adam optimizer (MAdam version 0.1) 
               to approximate a Pareto optimal "knee" solution. 
               Two objectives, the crossentropy loss and the l1 loss are used to re-interpret regularization from
               a multiobjective optimization point of view. For the stochastic multi gradient algorithms we refer 
               to the following paper for more details:

               S. Liu and L. N. Vicente, The stochastic multi-gradient algorithm for multi-objective optimization 
               and its application to supervised machine learning, ISE Technical Report 19T-011, Lehigh University.

               Not only vanilla SGD but also Keras version of Adam is extended to the multiobjective 
               case (for now only biobjective given). This is the main script for the (stochastic) dichotomic search 
               Pareto front plots. 
               For more information see our paper: 
               
               Reiners, M., Klamroth, K., Stiglmayr, M., 2020, Efficient and Sparse Neural Networks by Pruning 
               Weights in a Multiobjective Learning Approach, ARXIV LINK HERE

 Input(s)    : The MAdam Algorithm to solve the multiobjective optimization problem with two loss 
               functions, in this script it is combined with pruning neural networks weights. 
               Please note that for the (stochastic) dichotomic search the weighting of the loss functions remains constant over
               all training epochs (param multi=False) - there is no own calculation in the algorithm as suggested by
               S. Liu and L. N. Vicente. Only the dichotomic search algorihtm itself changes it. 
               Choose stochastic= boolean whether to use a stochastic dichotomic search or not. 

 Output(s)   : A Pareto optimal 'Knee' point for the trade-off between both objective functions (loss functions).
               Pruning success is involved. 
               Sparse, trained and pruned convolutional neural network architecture for the 'winning' weighting. 
               More details on the model architectures can be found in CustomModels.py.

 Notes       : The code is implemented using Python 3.7, Keras 2.3.1 and Tensorflow 1.14.0
               Please not that it is mandatory to use these versions of Tensorflow and Keras, otherwise the program 
               cannot be executed. The reason for this are the changed and adapted Keras and Tensorflow functions of 
               this particular versions.

'''

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
### Import Keras/Tensorflow Libaries
import tensorflow as tf
import keras
from keras import backend as K
from keras import regularizers
from keras.callbacks import LearningRateScheduler, LambdaCallback, Callback
from keras.engine.training import Model
from keras.optimizers import Optimizer, SGD, Adam, RMSprop
from keras.backend.tensorflow_backend import set_session

### Import Own Scripts (for multiobjctive optimization)
from MultiobjectiveOptimizers import SMGD, MAdam, MRMSprop
from MultiobjectiveClasses import Multi
from CustomLosses import L1loss, L2loss, L1lossDense, L2lossConv, L1L2lossDenseConv
from CustomModels import lenet5multimodel, lenet5regmodel, vggnetmultimodel, vggnetregmodel, get_data, update_weights

### Set up GPU ### private configs ## to be deleted 
tf_config= tf.ConfigProto()
os.environ['CUDA_VISIBLE_DEVICES']= '0'
tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
allow_soft_placement=True
tf_config.gpu_options.allow_growth= True
set_session(tf.Session(config=tf_config))

### Read in all information
learning_rate=float(sys.argv[1])
mnist= bool(True)
cifar10= bool(False)
stochastic= bool(True) # choose for a stochastic variant or the normal dichotomic search algorithm 
K.clear_session()

### Calcuation of the objective value for a given weigting: 'weight'
def calc_objective_value(weight, input_shape, x_train, y_train, x_test, y_test, num_classes, epochs):
    """ train and evaluate the network given specified weights
    :param:  - weight: weighting for the two objectives
             - input_shape: training data input shapes, e.g. amount of pixel
             - x_train: training data input (pictures)
             - y_train: training data groundtruth (classification number)
             - x_test: test data input (pictures)
             - y_test: test data groundtruth (classification number)
             - num_classes: amount of classes (e.g. 10 digits (0-9))
             - epochs: how long the training will last
    :return: - test_loss1: loss value on testing data (after training) from objective loss1 (e.g. Crossentropy)
             - test_loss2: loss value on testing data (after training) from objective loss2 (e.g. L1 loss)
             - L0ges: amount of nonzero weights in the dense layers of the model
             - L1ges: L1 values of all weights in the dense layers of the model
             - test_accuracy: accuracy value on the testing data
    """
    ### define and compile the model used for the training 
    weight_decay = 1e-4
    model = lenet5multimodel(input_shape=input_shape, weight_decay=weight_decay)
    model.mcompile(optimizer=MAdam(multi=False, learning_rate=learning_rate, descent_weight1=weight[0],descent_weight2=weight[1]),
                    loss1='sparse_categorical_crossentropy', loss2=L1lossDense(model),
                    metrics=['accuracy'])

    nonzero_weights1=[]
    nonzero_weights2=[]
    nonzero_weights3=[]

    ### perform a pruning step before the training starts
    weights1 = model.get_layer('denselayer1').get_weights()
    weights2 = model.get_layer('denselayer2').get_weights()
    weights3 = model.get_layer('denselayer3').get_weights() # weights and biases of last

    sparsified_weights1 = update_weights(weights1, 0.001)
    sparsified_weights2 = update_weights(weights2, 0.001)
    sparsified_weights3 = update_weights(weights3, 0.001)

    model.get_layer('denselayer1').set_weights(sparsified_weights1)
    model.get_layer('denselayer2').set_weights(sparsified_weights2)
    model.get_layer('denselayer3').set_weights(sparsified_weights3)

    nonzero_weights1.append([np.count_nonzero(
                        model.get_layer('denselayer1').get_weights()[0])])
    nonzero_weights2.append([np.count_nonzero(
                                model.get_layer('denselayer2').get_weights()[0])])
    nonzero_weights3.append([np.count_nonzero(
                                model.get_layer('denselayer3').get_weights()[0])])

    ### define a callback function that performs pruning after each iteration (after each batch)
    weight_callback_batch = LambdaCallback(on_batch_end=lambda batch,
                                                                logs: [
        model.get_layer(f"{name}").set_weights(update_weights(
            model.get_layer(f"{name}").get_weights(), 0.001))
        for name in ['denselayer1', 'denselayer2', 'denselayer3']])
    
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
    
    ### start the training process                                        
    history= model.mfit(x_train, y_train, epochs=epochs, validation_data=[x_test, y_test],
                                callbacks=[weight_callback_batch, safe_nonzeroweights1, safe_nonzeroweights2, safe_nonzeroweights3])

    ### calculate some information about the weights in the trained model 
    weights1 = model.get_layer("denselayer1").get_weights()  
    L1w1 = sum(sum(sum(np.abs(weights1))))
    L0w1 = np.count_nonzero(weights1[0]) + np.count_nonzero(weights1[1])

    weights2 = model.get_layer("denselayer2").get_weights()
    L1w2 = sum(sum(sum(np.abs(weights2))))
    L0w2 = np.count_nonzero(weights2[0]) + np.count_nonzero(weights2[1])

    weights3 = model.get_layer("denselayer3").get_weights()
    L1w3 = sum(sum(sum(np.abs(weights3))))
    L0w3 = np.count_nonzero(weights3[0]) + np.count_nonzero(weights3[1])

    L0ges= L0w1 + L0w2 + L0w3
    L1ges= L1w1 + L1w2 + L1w3
    ### evaluate the performance of the trained model on the test data 
    [test_loss1, test_loss2, test_accuracy] = model.evaluate_multi(x_test,y_test)

    return [test_loss1, test_loss2, L0ges,L1ges, test_accuracy, nonzero_weights1, nonzero_weights2, nonzero_weights3, history]

### Mark points in the convex hull 
def in_convex_hull(p, hull):
    """
    Test if points in `p` are in the convex hull
    :param: p: should be a `NxK` coordinates of `N` points in `K` dimensions
             hull: either a scipy.spatial.Delaunay object or the `MxK` array of the
             coordinates of `M` points in `K`dimensions for which Delaunay triangulation
             will be computed
    :return: boolean True or False, whether the point is in the convex hull

    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

### Mark pareto efficient solutions
def is_pareto_efficient(costs): 
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True
        if i> 0 and i < len(costs)-1 and in_convex_hull(c,np.asarray([costs[0], costs[-1], [costs[0][0],costs[0][1]*100], [costs[-1][0]*100,costs[-1][1]], [costs[0][0]*100,costs[0][1]*100] ])):
            is_efficient[i] = False
    return is_efficient

### Initialize (stochastic) dichotomic search--------------------------------------------------------------------------------------
chosen_lambdas=[]
x_train, y_train, x_test, y_test, train_data, input_shape, epochs, num_classes = get_data(mnist=mnist, cifar10=cifar10)
epochs_reduced=int(epochs/3)
level=int(15) # choose depth of the search tree and first weighting of the objectives
first_lambdas = [{'lambda': np.array([1., 0.])}, {'lambda': np.array([0.9, 0.1])}]
chosen_lambdas.append(np.array([1.,0.]))
chosen_lambdas.append(np.array([0.9,0.1]))
candidates: list = []  # the entries of the list will be dictionaries with key 'weight' and 'obj value'
candidates.extend(first_lambdas)
#### Calculate the first objective values for given weights------------------------------------------------------------
for candidate in candidates:
    # calculate objective value depending on the chosen weight
    f = calc_objective_value(candidate['lambda'], input_shape, x_train, y_train, x_test, y_test, num_classes,
                 epochs=epochs_reduced)
    # add to list
    candidate['obj value'] = np.array(f[:2])
candidates_history= candidates.copy() # collect all considered objective values
same=0
### Start the dichotomic search  --------------------------------------------------------------------------------------
for level in range(level):
    """
    list layout starting with values
         level 0: [A,                        B]
      -> level 1: [A,          A-B,          B]
      -> level 2: [A, A-(A-B), A-B, (A-B)-B, B]
      -> level n: [A, ..., A-(A-B), ..., A-B, ..., (A-B)-B, ..., B]
      and so on
    thus every second entry is new
    """
    print("%" * 30)
    print("level ", level+1)
    print("%" * 30)
    candidates.sort(key=lambda c: c['obj value'][1]) # sort by the second objective function
    i = 1  # start with the second entry
    signal=0 # plot the dichotomic search weighting or not (if it was a random choice not)
    while i < len(candidates):
        print(candidates)
        print("--- current list postion", i)
        # depending on two given objective values calculate a new weight l --------------------------------------------
        diff = candidates[i]['obj value'] - candidates[i - 1]['obj value']
        norm = LA.norm(diff, 1) # to normalize the weights
        if stochastic: 
            if norm > 0 and same <= 2: # only if the last two weightings are different enough
                diff_normalized = diff / norm
                l = np.array([diff_normalized[1], -diff_normalized[0]])  # new weight
            else:  # norm = 0, i.e. point is twice in the list or difference to the last objectives is not big enough
                l_0 = random.uniform(0.9*candidates[i]['lambda'][0], candidates[i]['lambda'][0])
                same=0 # resetting the counter
                l_1 = 1-l_0
                l = np.array([l_0, l_1])
                signal+=1 # indicate that last weight is randomly choosen --> no connections plotted in this level
        else: 
            if norm > 0: 
                diff_normalized = diff / norm
                l = np.array([diff_normalized[1], -diff_normalized[0]])  # new weight


        print("new weight:", l)
        chosen_lambdas.append(l)
        rounded = [float(np.round(x[1],4)) for x in chosen_lambdas]
        if rounded[-1] == rounded[-2]:
            same=1
            if rounded[-3]==rounded[-2]:
                same=2
                if rounded[-3]==rounded[-3]:
                    same=3
        else: 
            same=0        
        # calculate objective value depending on the chosen weight ------------------------------------------------------------------------------------
        f = calc_objective_value(l, input_shape, x_train, y_train, x_test, y_test, num_classes, epochs=epochs_reduced)
        # add to list between the two used objective values
        candidates.insert(i, {'lambda': l, 'obj value': np.array(f[:2])})

        if in_convex_hull(f[:2],np.asarray([ candidates[i]['obj value'], candidates[i - 1]['obj value'], [candidates[i]['obj value'][0],candidates[i - 1]['obj value'][1]] ])):
            expected+=1
        candidates_history.append({'lambda': l, 'obj value': np.array(f[:2])})
        # only every second entry is new
        i += 2

    print("current list of weighting vectors", chosen_lambdas)
    ### displaying after each level
    for candidate in candidates_history:
        hist,= plt.plot(candidate['obj value'][0], candidate['obj value'][1], 'x', color='black')
    for candidate in candidates: 
        curr, = plt.plot(candidate['obj value'][0], candidate['obj value'][1], 'x', color='green')

    plt.legend([hist,curr],['history','current'],loc='upper right')

    if len(candidates)>=3 and signal == 0:
        for i, candidate in enumerate(candidates):
            if i % 2 == 0:  # even numbers
                # plot old values
                plt.plot(candidate['obj value'][0], candidate['obj value'][1], 'o', color='orange', fillstyle='none')
            else:  # odd numbers
                # plot linking line between old values
                if 'obj value' in candidate:
                    diff = candidates[i+1]['obj value'] - candidates[i-1]['obj value']
                    norm = LA.norm(diff, 1)
                    x_plot = np.linspace(0, norm, 100)
                    plt.plot(-candidate['lambda'][1] * x_plot + candidates[i-1]['obj value'][0],
                            candidate['lambda'][0] * x_plot + candidates[i-1]['obj value'][1],
                            color='orange')
                    # plot new values
                    plt.plot(candidate['obj value'][0], candidate['obj value'][1], 'o', color='red', fillstyle='none')
    if stochastic:
        plt.title("Stochastic Dichotomic Search Level " + str(level+1))
    else: 
        plt.title("Dichotomic Search Level " + str(level+1))
    plt.xlabel('CE Loss on Test Data')
    plt.ylabel('L1 Loss on Test Data')
    plt.savefig(f"level:{level+1}lastweight:{l}{time.time()}-learning_rate{learning_rate}.png")
    plt.close()
    if len(candidates_history) >= 20: # upper limit for calculations
        break

    ### delete dominated after each level -------------------------------------------------------------------------------------------------------------
    efficient_candidates=[]
    for candidate in candidates:
        efficient_candidates.append(np.asarray(candidate['obj value']))
    efficient_candidates=np.asarray(efficient_candidates)
    mask= is_pareto_efficient(efficient_candidates)
    candidates= list(np.asarray(candidates)[mask])

### After the dichotomic search, choose the best lambda and validate it as knee solution  ------------------------------------------------------------  
lambdas = []
efficient_lambdas = []
for candidate in candidates:
    efficient_lambdas.append(np.asarray(candidate['lambda']))
for la in efficient_lambdas:
    if la[1] != 0:
        lambdas.append(la[0]/la[1])
diff= abs(lambdas- np.roll(lambdas,1))
lambda_ind= np.argmax(diff) -1
winning_lambda= efficient_lambdas[lambda_ind]
### Start the training on the whole epochs after chosen the winning lambda 
f = calc_objective_value(winning_lambda, input_shape, x_train, y_train, x_test, y_test, num_classes, epochs=epochs)

### Validate that the objectives are the most promising ones
k=0
for candidate in candidates_history:
    if candidate['obj value'][0] > f[0] or candidate['obj value'][1] > f[1]: 
        k+=1
if k == len(candidates_history):
    eff=True
else:
    eff=False

### Plot Accuracy
plt.plot(f[-1].history['accuracy'], 'b')
plt.plot(f[-1].history['val_accuracy'], 'g')
xs = np.linspace(1, 21, 35)
plt.hlines(y=0.989, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)
plt.title(f'Model Accuracy for Pareto Knee')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validate'], loc='lower right')
plt.savefig(f'Acc_mnist-LR-{learning_rate:.4}-{eff}.png')
plt.close()

### Plot Loss/Loss1
plt.plot(f[-1].history['loss1'], 'b')
plt.plot(f[-1].history['val_loss1'], 'g')
plt.title(f'Model Loss t={learning_rate:.4}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validate'], loc='upper right')
plt.savefig(f'Loss_mnist-LR-{learning_rate:.4}-{eff}.png')
plt.close()

### Plot Pruning Success of Layer1
plt.plot(f[-4])
xs = np.linspace(1, 21, 35)
plt.hlines(y=2107, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)
plt.title(f'Nonzero Weights Layer1')
plt.ylabel('Amount of Nonzeros Weights')
plt.xlabel('Epoch')
plt.savefig(f'Nonzeros1_mnist-LR-{learning_rate:.4}-pruning.png')
plt.close()

### Plot Pruning Success of Layer2
plt.plot(f[-3])
xs = np.linspace(1, 21, 35)
plt.hlines(y=300, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)
plt.axis([0, 35, 0, 11000])
plt.title(f'Nonzero Weights Layer2')
plt.ylabel('Amount of Nonzeros Weights')
plt.xlabel('Epoch')
plt.legend(['Train multi', 'Train SGD', 'Train Adam'], loc='upper right')
plt.savefig(f'Nonzeros2_mnist-LR-{learning_rate:.4}-pruning.png')
plt.close()

### Plot Pruning Success of Layer3
plt.plot(f[-2])
xs = np.linspace(1, 21, 35)
plt.hlines(y=170, xmin=0, xmax=len(xs), colors='0.5', linestyles='--', lw=2)
plt.title(f'Nonzero Weights Layer3')
plt.ylabel('Amount of Nonzeros Weights')
plt.xlabel('Epoch')
plt.legend(['Train multi', 'Train SGD', 'Train Adam'], loc='upper right')
plt.savefig(f'Nonzeros3_mnist-LR-{learning_rate:.4}-pruning.png')
plt.close()
