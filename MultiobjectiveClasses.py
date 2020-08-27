'''
 Author      :  Malena Reiners, M. Sc. Mathematics
 Description :  Provides a multiobjective optimization toolbox for Keras (Keras Extension from Version 2.3.1)
                Class inheritance of class Model (Multi) with adapted methods mcompile (multi loss compile), mfit (multi loss fit), 
                evaluate_multi,... and so on to handle multiple loss functions 

 Input(s)    :  A neural network architecture implemented as a Keras model with two loss function, see for example lenet5multimodel 
                in CustomModels.py

 Output(s)   :  Depending on the method, all outputs adapted for the multi objective optimization process.
                
                This is not a main file. 

 Notes       :  The code is implemented using Python 3.7, Keras 2.3.1 and Tensorflow 1.14.0
                Please not that it is mandatory to use these versions of Tensorflow and Keras, otherwise the program cannot be executed. 
                The reason for this are the changed and adapted Keras and Tensorflow functions of this particular versions. 

            
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

# ### custom scripts 
from CustomLosses import L1loss, L2loss, L1lossDense, L2lossConv, L1L2lossDenseConv

import numpy as np
if K.backend() == 'tensorflow':
    import tensorflow as tf

# print(tf.__version__) #1.14.0
# print(keras.__version__) #2.3.1

"""Multi Class Model written for Keras"""
class Multi(Model):
    """The `Multi` class is a subclass of the class 'Model' which adds training & evaluation routines to a `Network`.
        Multi loss functions can be used, only two different losses are supported for now.
    """

    def mcompile(self, optimizer,
                 loss1=None,
                 loss2=None,
                 metrics=None,
                 loss1_weights=None,
                 loss2_weights=None,
                 sample_weight_mode=None,
                 weighted_metrics=None,
                 target_tensors=None,
                 **kwargs):

        """Configures the Multi model for training.

              # Arguments
                  optimizer: String (name of optimizer) or optimizer instance.
                      See [optimizers](/optimizers).
                  loss1: String (name of objective function) or objective function or
                      `Loss` instance. See [losses](/losses).
                      If the model has multiple outputs, you can use a different loss
                      on each output by passing a dictionary or a list of losses.
                      The loss value that will be minimized by the model
                      will then be the sum of all individual losses.
                  loss2: String (name of objective function) or objective function or
                      `Loss` instance. See [losses](/losses).
                      If the model has multiple outputs, you can use a different loss
                      on each output by passing a dictionary or a list of losses.
                      The loss value that will be minimized by the model
                      will then be the sum of all individual losses.
                  metrics: List of metrics to be evaluated by the model
                      during training and testing. Typically you will use
                      `metrics=['accuracy']`. To specify different metrics for different
                      outputs of a multi-output model, you could also pass a dictionary,
                      such as
                      `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
                      You can also pass a list (len = len(outputs)) of lists of metrics
                      such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
                      `metrics=['accuracy', ['accuracy', 'mse']]`.
                  loss1_weights: Optional list or dictionary specifying scalar
                      coefficients (Python floats) to weight the loss1 contributions
                      of different model outputs.
                      The loss1 value that will be minimized by the model
                      will then be the *weighted sum* of all individual losses,
                      weighted by the `loss_weights` coefficients.
                      If a list, it is expected to have a 1:1 mapping
                      to the model's outputs. If a dict, it is expected to map
                      output names (strings) to scalar coefficients.
                  loss2_weights: Optional list or dictionary specifying scalar
                      coefficients (Python floats) to weight the loss2 contributions
                      of different model outputs.
                      The loss2 value that will be minimized by the model
                      will then be the *weighted sum* of all individual losses,
                      weighted by the `loss_weights` coefficients.
                      If a list, it is expected to have a 1:1 mapping
                      to the model's outputs. If a dict, it is expected to map
                      output names (strings) to scalar coefficients.
                  sample_weight_mode: If you need to do timestep-wise
                      sample weighting (2D weights), set this to `"temporal"`.
                      `None` defaults to sample-wise weights (1D).
                      If the model has multiple outputs, you can use a different
                      `sample_weight_mode` on each output by passing a
                      dictionary or a list of modes.
                  weighted_metrics: List of metrics to be evaluated and weighted
                      by sample_weight or class_weight during training and testing.
                  target_tensors: By default, Keras will create placeholders for the
                      model's target, which will be fed with the target data during
                      training. If instead you would like to use your own
                      target tensors (in turn, Keras will not expect external
                      Numpy data for these targets at training time), you
                      can specify them via the `target_tensors` argument. It can be
                      a single tensor (for a single-output model), a list of tensors,
                      or a dict mapping output names to target tensors.
                  **kwargs: When using the Theano/CNTK backends, these arguments
                      are passed into `K.function`.
                      When using the TensorFlow backend,
                      these arguments are passed into `tf.Session.run`.

              # Raises
                  ValueError: In case of invalid arguments for
                      `optimizer`, `loss1`, `loss2`, `metrics` or `sample_weight_mode`.
              """

        self.optimizer = optimizers.get(optimizer)
        self.loss1 = loss1 or {}
        self.loss2 = loss2 or {}
        self._compile_metrics = metrics or []
        self.loss1_weights = loss1_weights  # if one uses more than one "loss1" function give weights for weighted sum method
        self.loss2_weights = loss2_weights  # if one uses more than one "loss2" function give weights for weighted sum method
        self.sample_weight_mode = sample_weight_mode
        self._compile_weighted_metrics = weighted_metrics


        # List of stateful metric functions. Used for resetting metric state during
        # training/eval.
        self._compile_metric_functions = []
        # List of metric wrappers on output losses.
        self._output_loss_metrics = None

        if not self.built:
            # Model is not compilable because
            # it does not know its number of inputs
            # and outputs, nor their shapes and names.
            # We will compile after the first
            # time the model gets called on training data.
            return
        self._is_compiled = True

        # Prepare list of loss functions, same size as model outputs.
        self.loss_functions1 = training_utils.prepare_loss_functions(
            self.loss1, self.output_names)
        self.loss_functions2 = training_utils.prepare_loss_functions(
            self.loss2, self.output_names)
        
        ################### additional loss functions only in case of pruning and regularization important #################################
        ### for the option 'split' and only optional (for comparison reasons when regularization is included as a second objective function)
        self.loss_functions3 = training_utils.prepare_loss_functions(
           L1loss(self), self.output_names) #loss l1 
        self.loss_functions4 = training_utils.prepare_loss_functions(
           L2loss(self), self.output_names) #loss l2
        self.loss_functions5 = training_utils.prepare_loss_functions(
           L1lossDense(self), self.output_names) #loss l1 dense
        self.loss_functions6 = training_utils.prepare_loss_functions(
           L2lossConv(self), self.output_names) #loss l2 conv
        #####################################################################################################################################
        self._feed_outputs = []
        self._feed_output_names = []
        self._feed_output_shapes = []
        self._feed_loss1_fns = []
        self._feed_loss2_fns = []

        # if loss function1 is None, then this output will be skipped during total
        # loss calculation and feed targets preparation.
        self.skip_target_indices = []
        skip_target_weighing_indices = []
        for i, loss_function in enumerate(self.loss_functions1):
            if loss_function is None:
                self.skip_target_indices1.append(i)
                skip_target_weighing_indices.append(i)

        # Prepare output masks.
        masks = self.compute_mask(self.inputs, mask=None)
        if masks is None:
            masks = [None for _ in self.outputs]
        masks = to_list(masks)

        # Prepare list loss weights, same size of model outputs.
        self.loss_weights_list1 = training_utils.prepare_loss_weights(
            self.output_names, loss1_weights)  # used by prepare
        self.loss_weights_list2 = training_utils.prepare_loss_weights(
            self.output_names, loss2_weights)

        # Prepare targets of model.
        self.targets = []
        self._feed_targets = []
        if target_tensors is not None:
            if isinstance(target_tensors, list):
                if len(target_tensors) != len(self.outputs):
                    raise ValueError(
                        'When passing a list as `target_tensors`, '
                        'it should have one entry per model output. '
                        'The model has ' + str(len(self.outputs)) +
                        ' outputs, but you passed target_tensors=' +
                        str(target_tensors))
            elif isinstance(target_tensors, dict):
                for name in target_tensors:
                    if name not in self.output_names:
                        raise ValueError('Unknown entry in `target_tensors` '
                                         'dictionary: "' + name + '". '
                                                                  'Only expected the following keys: ' +
                                         str(self.output_names))
                tmp_target_tensors = []
                for name in self.output_names:
                    tmp_target_tensors.append(target_tensors.get(name, None))
                target_tensors = tmp_target_tensors
            elif K.is_tensor(target_tensors):
                if len(self.outputs) != 1:
                    raise ValueError('The model has ' + str(len(self.outputs)) +
                                     ' outputs, but you passed a single tensor as '
                                     '`target_tensors`. Expected a list or a dict '
                                     'of tensors.')
                target_tensors = [target_tensors]
            else:
                raise TypeError('Expected `target_tensors` to be a tensor, '
                                'a list of tensors, or dict of tensors, but got:',
                                target_tensors)

        for i in range(len(self.outputs)):
            if i in self.skip_target_indices:
                self.targets.append(None)
            else:
                shape = K.int_shape(self.outputs[i])
                name = self.output_names[i]
                if target_tensors is not None:
                    target = target_tensors[i]
                else:
                    target = None
                if target is None or K.is_placeholder(target):
                    if target is None:
                        target = K.placeholder(
                            ndim=len(shape),
                            name=name + '_target',
                            sparse=K.is_sparse(self.outputs[i]),
                            dtype=K.dtype(self.outputs[i]))
                    self._feed_targets.append(target)
                    self._feed_outputs.append(self.outputs[i])
                    self._feed_output_names.append(name)
                    self._feed_output_shapes.append(shape)
                    self._feed_loss1_fns.append(self.loss_functions1[i])
                    self._feed_loss2_fns.append(self.loss_functions2[i])
                else:
                    skip_target_weighing_indices.append(i)
                self.targets.append(target)

        # Prepare sample weights.
        self._set_sample_weight_attributes(
            sample_weight_mode, skip_target_weighing_indices)

        # Save all metric attributes per output of the model.
        self._cache_output_metric_attributes_multi(metrics, weighted_metrics)
        # Set metric attributes on model.
        self._set_metric_attributes_multi()

        # Invoke metric functions (unweighted) for all the outputs.
        self._handle_metrics(
            self.outputs,
            targets=self.targets,
            skip_target_masks=[l is None for l in self.loss_functions1],
            sample_weights=self.sample_weights,
            masks=masks)

        # Compute total loss.
        # Used to keep track of the total loss value (stateless)
        # eg., total_loss = loss_weight_1 * output_1_loss_fn(...) +
        #                   loss_weight_2 * output_2_loss_fn(...) +
        #                   layer losses.
        self.total_loss1 = self._prepare_total_losses(self.loss_functions1, self.loss_weights_list1,masks)
        self.total_loss2 = self._prepare_total_losses(self.loss_functions2, self.loss_weights_list2, masks)
        ################### additional loss functions only in case of pruning and regularization important #################################
        ### only for split option (regularization for different kind of layers)
        self.total_loss3 = self._prepare_total_losses(self.loss_functions3, self.loss_weights_list1, masks) #loss l1
        self.total_loss4 = self._prepare_total_losses(self.loss_functions4, self.loss_weights_list2, masks) #loss l2
        self.total_loss5 = self._prepare_total_losses(self.loss_functions5, self.loss_weights_list1, masks) #loss l1 dense
        self.total_loss6 = self._prepare_total_losses(self.loss_functions6, self.loss_weights_list2, masks) #loss l2 conv
        #####################################################################################################################################


        # Functions for train, test and predict will.
        # be compiled lazily when required.
        # This saves time when the user is not using all functions.
        self._function_kwargs = kwargs

        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # Collected trainable weights, sorted in topological order.
        trainable_weights = self.trainable_weights
        self._collected_trainable_weights = trainable_weights

    def metrics_names(self):
        """Returns the model's display labels for all outputs."""
        metric_names = ['loss1'] + ['loss2'] + ['c1'] + ['c2']
        if self._is_compiled:
            # Add output loss metric names to the metric names list.
            if len(self.outputs) > 1:
                metric_names.extend([
                    self.output_names[i] + '_loss'
                    for i in range(len(self.outputs))
                    if i not in self.skip_target_indices
                ])

            # Add compile metrics/weighted metrics' names to the metric names list.
            metric_names.extend([m.name for m in self._compile_metric_functions])

        # Add metric names from layers.
        for layer in self.layers:
            metric_names += [m.name for m in layer._metrics]
        metric_names += [m.name for m in self._metrics]
        return metric_names

    def _make_train_function_multi(self):
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        self._check_trainable_weights_consistency()
        if self.train_function is None:
            inputs = (self._feed_inputs +
                      self._feed_targets +
                      self._feed_sample_weights)
            if self._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]

            with K.name_scope('training'):
                with K.name_scope(self.optimizer.__class__.__name__):
                    training_updates, self.c1, self.c2 = self.optimizer.get_updates(
                        params=self._collected_trainable_weights,
                        loss1=self.total_loss1, loss2=self.total_loss2, loss3=self.total_loss3, loss4=self.total_loss4, loss5=self.total_loss5, loss6= self.total_loss6)
                updates = self.updates + training_updates
                metrics = self._get_training_eval_metrics()
                metrics_tensors = [
                    m._call_result for m in metrics if hasattr(m, '_call_result')
                ]
                metrics_updates = []
                for m in metrics:
                    metrics_updates.extend(m.updates)

                # Gets loss and metrics. Updates weights at each call.
                self.train_function = K.function(
                    inputs,
                    [self.total_loss1] + [self.total_loss2] + [self.c1] + [self.c2] + metrics_tensors,  # output
                    updates=updates + metrics_updates,
                    name='train_function',
                    **self._function_kwargs)

    def _make_test_function_multi(self):
        if not hasattr(self, 'test_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.test_function is None:
            inputs = (self._feed_inputs +
                      self._feed_targets +
                      self._feed_sample_weights)
            if self._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]

            metrics = self._get_training_eval_metrics()
            metrics_tensors = [
                m._call_result for m in metrics if hasattr(m, '_call_result')
            ]

            metrics_updates = []
            for m in metrics:
                if m is not 'c1' and m is not 'c2':
                    metrics_updates.extend(m.updates)

            # Return loss and metrics, no gradient updates.
            # Does update the network states.
            self.test_function = K.function(
                inputs,
                [self.total_loss1] + [self.total_loss2] + metrics_tensors,
                updates=self.state_updates + metrics_updates,
                name='test_function',
                **self._function_kwargs)

    def _standardize_user_data_multi(self, x,
                                     y=None,
                                     sample_weight=None,
                                     class_weight=None,
                                     check_array_lengths=True,
                                     batch_size=None):
        all_inputs = []
        if not self.built:
            # We need to use `x` to set the model inputs.
            # We type-check that `x` and `y` are either single arrays
            # or lists of arrays.
            if isinstance(x, (list, tuple)):
                if not all(isinstance(v, np.ndarray) or
                           K.is_tensor(v) for v in x):
                    raise ValueError('Please provide as model inputs '
                                     'either a single '
                                     'array or a list of arrays. '
                                     'You passed: x=' + str(x))
                all_inputs += list(x)
            elif isinstance(x, dict):
                raise ValueError('Please do not pass a dictionary '
                                 'as model inputs.')
            else:
                if not isinstance(x, np.ndarray) and not K.is_tensor(x):
                    raise ValueError('Please provide as model inputs '
                                     'either a single '
                                     'array or a list of arrays. '
                                     'You passed: x=' + str(x))
                all_inputs.append(x)

            # Build the model using the retrieved inputs (value or symbolic).
            # If values, then in symbolic-mode placeholders will be created
            # to match the value shapes.
            if not self.inputs:
                self._set_inputs(x)

        if y is not None:
            if not self.optimizer:
                raise RuntimeError('You must compile a model before '
                                   'training/testing. '
                                   'Use `model.compile(optimizer, loss)`.')
            if not self._is_compiled:
                # On-the-fly compilation of the model.
                # We need to use `y` to set the model targets.
                if isinstance(y, (list, tuple)):
                    if not all(isinstance(v, np.ndarray) or
                               K.is_tensor(v) for v in y):
                        raise ValueError('Please provide as model targets '
                                         'either a single '
                                         'array or a list of arrays. '
                                         'You passed: y=' + str(y))
                elif isinstance(y, dict):
                    raise ValueError('Please do not pass a dictionary '
                                     'as model targets.')
                else:
                    if not isinstance(y, np.ndarray) and not K.is_tensor(y):
                        raise ValueError('Please provide as model targets '
                                         'either a single '
                                         'array or a list of arrays. '
                                         'You passed: y=' + str(y))
                # Typecheck that all inputs are *either* value *or* symbolic.
                if y is not None:
                    all_inputs += to_list(y, allow_tuple=True)
                if any(K.is_tensor(v) for v in all_inputs):
                    if not all(K.is_tensor(v) for v in all_inputs):
                        raise ValueError('Do not pass inputs that mix Numpy '
                                         'arrays and symbolic tensors. '
                                         'You passed: x=' + str(x) +
                                         '; y=' + str(y))

                # Handle target tensors if any passed.
                y = to_list(y, allow_tuple=True)
                target_tensors = [v for v in y if K.is_tensor(v)]
                if not target_tensors:
                    target_tensors = None
                self.mcompile(optimizer=self.optimizer,
                              loss1=self.loss1,
                              loss2=self.loss2,
                              metrics=self._compile_metrics,
                              weighted_metrics=self._compile_weighted_metrics,
                              loss1_weights=self.loss1_weights,
                              loss2_weights=self.loss2_weights,
                              target_tensors=target_tensors)

        # If `x` and `y` were all symbolic,
        # then the model should not be fed any inputs and targets.
        # Note: in this case, `any` and `all` are equivalent since we disallow
        # mixed symbolic/value inputs.
        if any(K.is_tensor(v) for v in all_inputs):
            return [], [], []

        # What follows is input validation and standardization to list format,
        # in the case where all inputs are value arrays.

        if not self._is_graph_network:
            # Case: symbolic-mode subclassed network.
            # Do not do shape validation.
            feed_input_names = self._feed_input_names
            feed_input_shapes = None
        else:
            # Case: symbolic-mode graph network.
            # In this case, we run extensive shape validation checks.
            feed_input_names = self._feed_input_names
            feed_input_shapes = self._feed_input_shapes

        # Standardize the inputs.
        x = training_utils.standardize_input_data(
            x,
            feed_input_names,
            feed_input_shapes,
            check_batch_axis=False,  # Don't enforce the batch size.
            exception_prefix='input')

        if y is not None:
            if not self._is_graph_network:
                feed_output_names = self._feed_output_names
                feed_output_shapes = None
                # Sample weighting not supported in this case.
                # TODO: consider supporting it.
                feed_sample_weight_modes = [None for _ in self.outputs]
            else:
                feed_output_names = self._feed_output_names
                feed_sample_weight_modes = self._feed_sample_weight_modes
                feed_output_shapes = []
                for output_shape, loss1_fn, loss2_fn in zip(self._feed_output_shapes,
                                                            self._feed_loss1_fns, self._feed_loss2_fns):
                    if ((isinstance(loss1_fn, losses.LossFunctionWrapper) and
                         loss1_fn.fn == losses.sparse_categorical_crossentropy)) or (
                            isinstance(
                                loss1_fn,
                                losses.SparseCategoricalCrossentropy)): 
                        if K.image_data_format() == 'channels_first' and len(
                                output_shape) in [4, 5]:
                            feed_output_shapes.append(
                                (output_shape[0], 1) + output_shape[2:])
                        else:
                            feed_output_shapes.append(output_shape[:-1] + (1,))
                    elif (not isinstance(loss1_fn, losses.Loss) or
                          (isinstance(loss1_fn, losses.LossFunctionWrapper) and
                           (getattr(losses, loss1_fn.fn.__name__, None) is None))):
                        # If the given loss is not an instance of the `Loss` class
                        # (custom class) or if the loss function that is wrapped is
                        # not in the `losses` module, then it is a user-defined loss
                        # and we make no assumptions about it.
                        feed_output_shapes.append(None)
                    else:
                        feed_output_shapes.append(output_shape)

                    if ((isinstance(loss2_fn, losses.LossFunctionWrapper) and
                         loss2_fn.fn == losses.sparse_categorical_crossentropy)) or (
                            isinstance(
                                loss2_fn, losses.SparseCategoricalCrossentropy)):
                        if K.image_data_format() == 'channels_first' and len(
                                output_shape) in [4, 5]:
                            feed_output_shapes.append(
                                (output_shape[0], 1) + output_shape[2:])
                        else:
                            feed_output_shapes.append(output_shape[:-1] + (1,))
                    elif (not isinstance(loss2_fn, losses.Loss) or
                          (isinstance(loss2_fn, losses.LossFunctionWrapper) and
                           (getattr(losses, loss2_fn.fn.__name__, None) is None))):
                        # If the given loss is not an instance of the `Loss` class
                        # (custom class) or if the loss function that is wrapped is
                        # not in the `losses` module, then it is a user-defined loss
                        # and we make no assumptions about it.
                        feed_output_shapes.append(None)
                    else:
                        feed_output_shapes.append(output_shape)

            # Standardize the outputs.
            y = training_utils.standardize_input_data(
                y,
                feed_output_names,
                feed_output_shapes,
                check_batch_axis=False,  # Don't enforce the batch size.
                exception_prefix='target')

            # Generate sample-wise weight values given the `sample_weight` and
            # `class_weight` arguments.
            sample_weights = training_utils.standardize_sample_weights(
                sample_weight, feed_output_names)
            class_weights = training_utils.standardize_class_weights(
                class_weight, feed_output_names)
            sample_weights = [
                training_utils.standardize_weights(ref, sw, cw, mode)
                for (ref, sw, cw, mode) in
                zip(y, sample_weights, class_weights,
                    feed_sample_weight_modes)
            ]
            # Check that all arrays have the same length.
            if check_array_lengths:
                training_utils.check_array_length_consistency(x, y, sample_weights)
            if self._is_graph_network:
                # Additional checks to avoid users mistakenly
                # using improper loss fns.
                training_utils.check_loss_and_target_compatibility(
                    y, self._feed_loss1_fns, feed_output_shapes)
                training_utils.check_loss_and_target_compatibility(
                    y, self._feed_loss2_fns, feed_output_shapes)
        else:
            y = []
            sample_weights = []

        if self.stateful and batch_size:
            # Check that for stateful networks, number of samples is a multiple
            # of the static batch size.
            if x[0].shape[0] % batch_size != 0:
                raise ValueError('In a stateful network, '
                                 'you should only pass inputs with '
                                 'a number of samples that can be '
                                 'divided by the batch size. Found: ' +
                                 str(x[0].shape[0]) + ' samples')
        return x, y, sample_weights

    def _prepare_total_losses(self, loss_functions, loss_weights_list, masks=None):
        """Computes total loss from loss functions.

        # Arguments
            skip_target_indices: A list of indices of model outputs where loss
                function is None.
            masks: List of mask values corresponding to each model output.

        # Returns
            A list of loss weights of python floats.
        """
        total_loss = None
        with K.name_scope('loss'):
            zipped_inputs = zip(self.targets, self.outputs, loss_functions,
                                self.sample_weights, masks, loss_weights_list)
            for i, (y_true, y_pred, loss_fn, sample_weight, mask,
                    loss_weight) in enumerate(zipped_inputs):
                if i in self.skip_target_indices:
                    continue
                loss_name = self.output_names[i] + '_loss'  # only for different output layers relevant
                with K.name_scope(loss_name):
                    if mask is not None:
                        mask = K.cast(mask, y_pred.dtype)
                        # Update weights with mask.
                        if sample_weight is None:
                            sample_weight = mask
                        else:
                            # Update dimensions of weights to match with mask.
                            mask, _, sample_weight = (
                                losses_utils.squeeze_or_expand_dimensions(
                                    mask, None, sample_weight))
                            sample_weight *= mask

                    output_loss = loss_fn(
                        y_true, y_pred, sample_weight=sample_weight)

                if len(self.outputs) > 1:
                    update_ops = self._output_loss_metrics[i].update_state(
                        output_loss)
                    with K.control_dependencies(update_ops):  # For TF
                        self._output_loss_metrics[i].result()
                if total_loss is None:
                    total_loss = loss_weight * output_loss
                else:
                    total_loss += loss_weight * output_loss

            if total_loss is None:
                if not self.losses:
                    raise ValueError('The model cannot be compiled '
                                     'because it has no loss to optimize.')
                else:
                    total_loss = 0.

            # Add regularization penalties and other layer-specific losses. # not used when split option is used
            if not self.optimizer.split:
                for loss_tensor in self.losses:
                    total_loss += loss_tensor

        return K.mean(total_loss)

    def _add_unique_metric_name_multi(self, metric_name, output_index):
        """Makes the metric name unique and adds it to the model's metric name list.

        If there are multiple outputs for which the metrics are calculated, the
        metric names have to be made unique by appending an integer.

        # Arguments
            metric_name: Metric name that corresponds to the metric specified by the
                user. For example: 'acc'.
            output_index: The index of the model output for which the metric name is
                being added.

        # Returns
            string, name of the model's unique metric name
        """
        if len(self.output_names) > 1:
            metric_name = '%s_%s' % (self.output_names[output_index], metric_name)

        j = 1
        base_metric_name = metric_name
        self.metric_names = self.metrics_names() ## added, as otherwise self.metric_names does not exist
        while metric_name in self.metric_names:
            metric_name = '%s_%d' % (base_metric_name, j)
            j += 1
        return metric_name

    def _set_metric_attributes_multi(self):
        """Sets the metric attributes on the model for all the model outputs."""
        updated_per_output_metrics = []
        updated_per_output_weighted_metrics = []
        for i in range(len(self.outputs)):
            if i in self.skip_target_indices:
                updated_per_output_metrics.append(self._per_output_metrics[i])
                updated_per_output_weighted_metrics.append(
                    self._per_output_weighted_metrics[i])
                continue
            updated_per_output_metrics.append(
                self._set_per_output_metric_attributes_multi(
                    self._per_output_metrics[i], i))
            updated_per_output_weighted_metrics.append(
                self._set_per_output_metric_attributes_multi(
                    self._per_output_weighted_metrics[i], i))

        # Create a metric wrapper for each output loss. This computes mean of an
        # output loss across mini-batches (irrespective of how we reduce within a
        # batch).
        if len(self.outputs) > 1:
            self._output_loss_metrics = [
                metrics_module.Mean(name=self.output_names[i] + '_loss')
                for i in range(len(self.loss_functions1))
            ]

        self._per_output_metrics = updated_per_output_metrics
        self._per_output_weighted_metrics = updated_per_output_weighted_metrics

    def _cache_output_metric_attributes_multi(self, metrics, weighted_metrics):
        """Caches metric name and function attributes for every model output."""
        output_shapes = []
        for output in self.outputs:
            if output is None:
                output_shapes.append(None)
            else:
                output_shapes.append(list(output.shape))
        self._per_output_metrics = training_utils.collect_per_output_metric_info(
            metrics, self.output_names, output_shapes, self.loss_functions1)
        self._per_output_weighted_metrics = (
            training_utils.collect_per_output_metric_info(
                weighted_metrics,
                self.output_names,
                output_shapes,
                self.loss_functions1,
                is_weighted=True))

    def _set_per_output_metric_attributes_multi(self, metrics_dict, output_index):
        """Sets the metric attributes on the model for the given output.

        # Arguments
            metrics_dict: A dict with metric names as keys and metric fns as values.
            output_index: The index of the model output for which the metric
                attributes are added.

        # Returns
            Metrics dict updated with unique metric names as keys.
        """
        updated_metrics_dict = collections.OrderedDict()
        for metric_name, metric_fn in metrics_dict.items():
            metric_name = self._add_unique_metric_name_multi(metric_name, output_index)

            # Update the name on the metric class to be the unique generated name.
            metric_fn.name = metric_name
            updated_metrics_dict[metric_name] = metric_fn
            # Keep track of metric function.
            self._compile_metric_functions.append(metric_fn)
        return updated_metrics_dict

    def mfit(self,
             x=None,
             y=None,
             batch_size=None,
             epochs=1,
             verbose=1,
             callbacks=None,
             validation_split=0.,
             validation_data=None,
             shuffle=True,
             class_weight=None,
             sample_weight=None,
             initial_epoch=0,
             steps_per_epoch=None,
             validation_steps=None,
             validation_freq=1,
             max_queue_size=10,
             workers=1,
             use_multiprocessing=False,
             **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        # Arguments
            x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                  (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding
                  array/tensors, if the model has named inputs.
                - A generator or `keras.utils.Sequence` returning
                  `(inputs, targets)` or `(inputs, targets, sample weights)`.
                - None (default) if feeding from framework-native
                  tensors (e.g. TensorFlow data tensors).
            y: Target data. Like the input data `x`,
                it could be either Numpy array(s), framework-native tensor(s),
                list of Numpy arrays (if the model has multiple outputs) or
                None (default) if feeding from framework-native tensors
                (e.g. TensorFlow data tensors).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                If `x` is a generator, or `keras.utils.Sequence` instance,
                `y` should not be specified (since targets will be obtained
                from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of symbolic tensors, generators, or `Sequence` instances
                (since they generate batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training and validation
                (if ).
                See [callbacks](/callbacks).
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
                This argument is not supported when `x` is a generator or
                `Sequence` instance.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                    - tuple `(x_val, y_val)` of Numpy arrays or tensors
                    - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
                    - dataset or a dataset iterator
                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` must be provided.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`. This argument
                is not supported when `x` generator, or `Sequence` instance,
                instead provide the sample_weights as the third element of `x`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
            validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.
            validation_steps: Only relevant if `validation_data` is provided
                and is a generator. Total number of steps (batches of samples)
                to draw before stopping when performing validation at the end
                of every epoch.
            validation_freq: Only relevant if validation data is provided. Integer
                or list/tuple/set. If an integer, specifies how many training
                epochs to run before a new validation run is performed, e.g.
                `validation_freq=2` runs validation every 2 epochs. If a list,
                tuple, or set, specifies the epochs on which to run validation,
                e.g. `validation_freq=[1, 2, 10]` runs validation at the end
                of the 1st, 2nd, and 10th epochs.
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1. If 0, will execute the generator on the main
                thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
            **kwargs: Used for backwards compatibility.

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        # Legacy support
        if 'nb_epoch' in kwargs:
            warnings.warn('The `nb_epoch` argument in `fit` '
                          'has been renamed `epochs`.', stacklevel=2)
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        if x is None and y is None and steps_per_epoch is None:
            raise ValueError('If fitting from data tensors, '
                             'you should specify the `steps_per_epoch` '
                             'argument.')

        batch_size = self._validate_or_infer_batch_size(
            batch_size, steps_per_epoch, x)

        # Case 1: generator-like. Input is Python generator,
        # or Sequence object, or iterator.
        if training_utils.is_generator_or_sequence(x):
            training_utils.check_generator_arguments(
                y, sample_weight, validation_split=validation_split)
            return self.fit_generator(
                x,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                validation_steps=validation_steps,
                validation_freq=validation_freq,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                shuffle=shuffle,
                initial_epoch=initial_epoch)

        # Case 2: Symbolic tensors or Numpy array-like.
        x, y, sample_weights = self._standardize_user_data_multi(
            x, y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            batch_size=batch_size)

        # Prepare validation data.
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('When passing validation_data, '
                                 'it must contain 2 (x_val, y_val) '
                                 'or 3 (x_val, y_val, val_sample_weights) '
                                 'items, however it contains %d items' %
                                 len(validation_data))

            val_x, val_y, val_sample_weights = self._standardize_user_data_multi(
                val_x, val_y,
                sample_weight=val_sample_weight,
                batch_size=batch_size)
            if self._uses_dynamic_learning_phase():
                val_inputs = val_x + val_y + val_sample_weights + [0]
            else:
                val_inputs = val_x + val_y + val_sample_weights

        elif validation_split and 0. < validation_split < 1.:
            if any(K.is_tensor(t) for t in x):
                raise ValueError(
                    'If your data is in the form of symbolic tensors, '
                    'you cannot use `validation_split`.')
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(int(x[0].shape[0]) * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
            sample_weights, val_sample_weights = (
                slice_arrays(sample_weights, 0, split_at),
                slice_arrays(sample_weights, split_at))
            if self._uses_dynamic_learning_phase():
                val_inputs = val_x + val_y + val_sample_weights + [0]
            else:
                val_inputs = val_x + val_y + val_sample_weights

        elif validation_steps:
            do_validation = True
            if self._uses_dynamic_learning_phase():
                val_inputs = [0.]

        # Prepare input arrays and training function.
        if self._uses_dynamic_learning_phase():
            fit_inputs = x + y + sample_weights + [1]
        else:
            fit_inputs = x + y + sample_weights
        self._make_train_function_multi()
        fit_function = self.train_function

        # Prepare display labels.fit_loop_multilo
        out_labels = self.metrics_names() 
        self.metric_names = self.metrics_names()

        if do_validation:
            self._make_test_function_multi()
            val_function = self.test_function
        else:
            val_function = None
            val_inputs = []

        # Delegate logic to `fit_loop`.
        return fit_loop_multi(self, fit_function, fit_inputs,
                                  out_labels=out_labels,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=verbose,
                                  callbacks=callbacks,
                                  val_function=val_function,
                                  val_inputs=val_inputs,
                                  shuffle=shuffle,
                                  initial_epoch=initial_epoch,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  validation_freq=validation_freq)


    def evaluate_multi(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False):
        """Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches.

        # Arguments
            x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                  (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding
                  array/tensors, if the model has named inputs.
                - A generator or `keras.utils.Sequence` returning
                  `(inputs, targets)` or `(inputs, targets, sample weights)`.
                - None (default) if feeding from framework-native
                  tensors (e.g. TensorFlow data tensors).
            y: Target data. Like the input data `x`,
                it could be either Numpy array(s), framework-native tensor(s),
                list of Numpy arrays (if the model has multiple outputs) or
                None (default) if feeding from framework-native tensors
                (e.g. TensorFlow data tensors).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                If `x` is a generator, or `keras.utils.Sequence` instance,
                `y` should not be specified (since targets will be obtained
                from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of symbolic tensors, generators, or
                `keras.utils.Sequence` instances (since they generate batches).
            verbose: 0 or 1. Verbosity mode.
                0 = silent, 1 = progress bar.
            sample_weight: Optional Numpy array of weights for
                the test samples, used for weighting the loss function.
                You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
                See [callbacks](/callbacks).
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up when using
                process-based threading. If unspecified, `workers` will default
                to 1. If 0, will execute the generator on the main thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.

        # Raises
            ValueError: in case of invalid arguments.

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """

        batch_size = self._validate_or_infer_batch_size(batch_size, steps, x)

        # Case 1: generator-like. Input is Python generator, or Sequence object.
        if training_utils.is_generator_or_sequence(x):
            training_utils.check_generator_arguments(y, sample_weight)
            return self.evaluate_generator(
                x,
                steps=steps,
                verbose=verbose,
                callbacks=callbacks,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing)

        # Case 2: Symbolic tensors or Numpy array-like.
        if x is None and y is None and steps is None:
            raise ValueError('If evaluating from data tensors, '
                             'you should specify the `steps` '
                             'argument.')
        # Validate user data.
        x, y, sample_weights = self._standardize_user_data_multi(
            x, y,
            sample_weight=sample_weight,
            batch_size=batch_size)
        # Prepare inputs, delegate logic to `test_loop`.
        if self._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [0]
        else:
            ins = x + y + sample_weights
        self._make_test_function()
        f = self.test_function
        return test_loop_multi(self, f, ins,
                        batch_size=batch_size,
                        verbose=verbose,
                        steps=steps,
                        callbacks=callbacks)


"""Fit Loop Multi Loss written for Keras"""
def fit_loop_multi(model, fit_function, fit_inputs,
                       out_labels=None,
                       batch_size=None,
                       epochs=100,
                       verbose=1,
                       callbacks=None,
                       val_function=None,
                       val_inputs=None,
                       shuffle=True,
                       initial_epoch=0,
                       steps_per_epoch=None,
                       validation_steps=None,
                       validation_freq=1):
    """Abstract fit function for `fit_function(fit_inputs)`.

    Assumes that fit_function returns a list, labeled by out_labels.

    # Arguments
        model: Keras model instance.
        fit_function: Keras function returning a list of tensors
        fit_inputs: List of tensors to be fed to `fit_function`
        out_labels: List of strings, display names of
            the outputs of `fit_function`
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training and validation
            (if `val_function` and `val_inputs` are not `None`).
        val_function: Keras function to call for validation
        val_inputs: List of tensors to be fed to `val_function`
        shuffle: Whether to shuffle the data at the beginning of each epoch
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.
        validation_freq: Only relevant if validation data is provided. Integer
            or list/tuple/set. If an integer, specifies how many training
            epochs to run before a new validation run is performed, e.g.
            validation_freq=2` runs validation every 2 epochs. If a list,
            tuple, or set, specifies the epochs on which to run validation,
            e.g. `validation_freq=[1, 2, 10]` runs validation at the end
            of the 1st, 2nd, and 10th epochs.

    # Returns
        `History` object.
    """
    do_validation = False
    if val_function and val_inputs:
        do_validation = True
        if (verbose and fit_inputs and
                hasattr(fit_inputs[0], 'shape') and hasattr(val_inputs[0], 'shape')):
            print('Train on %d samples, validate on %d samples' %
                  (fit_inputs[0].shape[0], val_inputs[0].shape[0]))
    if validation_steps:
        do_validation = True
        if steps_per_epoch is None:
            raise ValueError('Can only use `validation_steps` '
                             'when doing step-wise '
                             'training, i.e. `steps_per_epoch` '
                             'must be set.')
    elif do_validation:
        if steps_per_epoch:
            raise ValueError('Must specify `validation_steps` '
                             'to perform validation '
                             'when doing step-wise training.')

    num_train_samples = check_num_samples(fit_inputs,
                                          batch_size=batch_size,
                                          steps=steps_per_epoch,
                                          steps_name='steps_per_epoch')
    if num_train_samples is not None:
        index_array = np.arange(num_train_samples)

    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(stateful_metrics=model.metric_names[4:])]
    if verbose:
        if steps_per_epoch is not None:
            count_mode = 'steps'
        else:
            count_mode = 'samples'
        _callbacks.append(
            cbks.ProgbarLogger(count_mode, stateful_metrics=model.metric_names[4:]))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)
    out_labels = out_labels or []

    # it's possible to callback a different model than itself
    # (used by Sequential models)
    callback_model = model._get_callback_model()
    callback_metrics = list(model.metric_names)
    metrics= model.metric_names[:2]+model.metric_names[4:]
    if do_validation:
        callback_metrics += ['val_' + n for n in metrics]

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks._call_begin_hook('train')
    callbacks.model.stop_training = False
    for cbk in callbacks:
        cbk.validation_data = val_inputs

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion.
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(fit_inputs[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}
        if steps_per_epoch is not None:
            for step_index in range(steps_per_epoch):
                batch_logs = {'batch': step_index, 'size': 1}
                callbacks._call_batch_hook('train', 'begin', step_index, batch_logs)
                outs = fit_function(fit_inputs)

                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks._call_batch_hook('train', 'end', step_index, batch_logs)
                if callback_model.stop_training:
                    break

            if do_validation and should_run_validation(validation_freq, epoch):
                val_outs = test_loop_multi(model, val_function, val_inputs,
                                     steps=validation_steps,
                                     callbacks=callbacks,
                                     verbose=0)
                val_outs = to_list(val_outs)
                # Same labels assumed
                for l, o in zip(metrics, val_outs):
                    epoch_logs['val_' + l] = o
        else:
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(num_train_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(fit_inputs[-1], int):
                        # Do not slice the training phase flag.
                        ins_batch = slice_arrays(
                            fit_inputs[:-1], batch_ids) + [fit_inputs[-1]]
                    else:
                        ins_batch = slice_arrays(fit_inputs, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {'batch': batch_index, 'size': len(batch_ids)}
                callbacks._call_batch_hook('train', 'begin', batch_index, batch_logs)
                for i in indices_for_conversion_to_dense:
                    ins_batch[i] = ins_batch[i].toarray()

                outs = fit_function(ins_batch)
                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)
                if callbacks.model.stop_training:
                    break

            if batch_index == len(batches) - 1:  # Last batch.
                if do_validation and should_run_validation(validation_freq, epoch):
                    val_outs = test_loop_multi(model, val_function, val_inputs,
                                                         batch_size=batch_size,
                                                         callbacks=callbacks,
                                                         verbose=0)
                    val_outs = to_list(val_outs)
                    # Same labels assumed.
                    for l, o in zip(metrics, val_outs):
                        epoch_logs['val_' + l] = o

        callbacks.on_epoch_end(epoch, epoch_logs)
        if callbacks.model.stop_training:
            break
    callbacks._call_end_hook('train')
    return model.history

"""Test Loop Multi written for Keras"""
def test_loop_multi(model, f, ins,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None):
    """Abstract method to loop over some data in batches.

    # Arguments
        model: Keras model instance.
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size or `None`.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring predictions finished.
            Ignored with the default value of `None`.
        callbacks: List of callbacks or an instance of
            `keras.callbacks.CallbackList` to be called during evaluation.

    # Returns
        Scalar loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """

    model.reset_metrics()
    num_samples = check_num_samples(ins,
                                    batch_size=batch_size,
                                    steps=steps,
                                    steps_name='steps')

    # Check if callbacks have not been already configured
    if not isinstance(callbacks, cbks.CallbackList):
        callbacks = cbks.CallbackList(callbacks)
        callback_model = model._get_callback_model()
        callbacks.set_model(callback_model)
        callback_metrics = list(model.metric_names)
        callback_params = {
            'batch_size': batch_size,
            'steps': steps,
            'samples': num_samples,
            'verbose': verbose,
            'metrics': callback_metrics,
        }
        callbacks.set_params(callback_params)

    outs = []
    if verbose == 1:
        if steps is not None:
            progbar = Progbar(target=steps)
        else:
            progbar = Progbar(target=num_samples)

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion.
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(ins[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    callbacks.model.stop_training = False
    callbacks._call_begin_hook('test')

    if steps is not None:
        for step in range(steps):
            batch_logs = {'batch': step, 'size': 1}
            callbacks._call_batch_hook('test', 'begin', step, batch_logs)
            batch_outs = f(ins)
            if isinstance(batch_outs, list):
                if step == 0:
                    outs.extend([0.] * len(batch_outs))
                for i, batch_out in enumerate(batch_outs):
                    if i == 0 or i == 1:  # Index 0 == `loss1`, Index 1 == `loss2`
                        outs[i] = float(batch_out)
                    else:
                        outs[i] += float(batch_out)
            else:
                if step == 0:
                    outs.append(0.)
                outs[0] += float(batch_outs)

            for l, o in zip(model.metric_names, batch_outs):
                batch_logs[l] = o
            callbacks._call_batch_hook('test', 'end', step, batch_logs)

            if verbose == 1:
                progbar.update(step + 1)
        outs[0] /= steps  # Index 0 == `Loss`
        outs[1] /= steps
    else:
        batches = make_batches(num_samples, batch_size)
        index_array = np.arange(num_samples)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if isinstance(ins[-1], int):
                # Do not slice the training phase flag.
                ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_arrays(ins, batch_ids)
            for i in indices_for_conversion_to_dense:
                ins_batch[i] = ins_batch[i].toarray()

            batch_logs = {'batch': batch_index, 'size': len(batch_ids)}
            callbacks._call_batch_hook('test', 'begin', batch_index, batch_logs)
            batch_outs = f(ins_batch)
            if isinstance(batch_outs, list):
                if batch_index == 0:
                    outs.extend([0.] * len(batch_outs))
                for i, batch_out in enumerate(batch_outs):
                    if i == 0 or i ==1:  # Index 0 == `loss1`, Index 1 == `loss2`
                        outs[i] += float(batch_out) * len(batch_ids)
                    else:
                        outs[i] = float(batch_out)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += float(batch_outs) * len(batch_ids)

            metrics= model.metric_names[:2]+ model.metric_names[4:] #new

            for l, o in zip(metrics, batch_outs):
                batch_logs[l] = float(o)
            callbacks._call_batch_hook('test', 'end', batch_index, batch_logs)

            if verbose == 1:
                progbar.update(batch_end)
        outs[0] /= num_samples  # Index 0 == `loss1`
        outs[1] /= num_samples # Index 1 == `loss2`
    callbacks._call_end_hook('test')
    return unpack_singleton(outs)