import importlib
from imp import reload

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn

# import f_df_mlp_vislecun as fdf
# reload(fdf)
import timeit
import time

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.regularization import *

FLOATX='float32' # needed to use the GPU

GRAD_CLIP = 100
DEFAULT_BATCH_PERC = 0.1
DEFAULT_BATCH_SIZE = 7000

class Optimizer(object):


    def __init__(self, predictive_network,verbose = 1):

        self.predictive_network = predictive_network
        self.train_fn = predictive_network.train_fn
        self.val_fn = predictive_network.val_fn
        #self.batch_size = predictive_network.batch_size
        self.batch_size = DEFAULT_BATCH_SIZE
        self.verbose = verbose
        self._orig_start_time = None
        self._num_epochs_run = 0
        self.cost_history = {'train_costs': [],
                             'val_costs': [],
                             'final_train_cost': 0,
                             'final_val_cost': 0}



    #-------------------------------------------------------------------------------
    #-----------------------------Update functions----------------------------------
    #-------------------------------------------------------------------------------

    def run_all_epochs(self, X_train, y_train, X_val=None, y_val=None, num_epochs=100):
        # train_costs = []
        # val_costs = []
        
        if num_epochs is not None:
            self.run_n_epochs(self, X_train, y_train, X_val=X_val, y_val=y_val, num_epochs=num_epochs)
        # else:
        #     self._run_epochs_till_converge(self, X_train, y_train, X_val=None, y_val=None)

        return 



    def run_n_epochs(self, X_train, y_train, X_val, y_val, num_epochs):
        

        for epoch in range(num_epochs):
            start_time = time.time()

            if X_val is not None:
                val_err = self._run_an_epoch(X_val, y_val, self.val_fn)
                self.cost_history['val_costs'].append(val_err)
            else:
                val_err = None

            train_err = self._run_an_epoch(X_train, y_train, self.train_fn)
            self.cost_history['train_costs'].append(train_err)

            if self.verbose: 
                print_progress(epoch, num_epochs, train_err, start_time,
                               val_err=val_err, 
                               final_epoch=False, 
                               print_period=np.minimum(int(num_epochs/10),100))

            if epoch == num_epochs - 1:
                print_progress(epoch, num_epochs, train_err, self._orig_start_time,
                                    val_err=val_err,
                                    final_epoch=True,
                                    print_period=1)

            # self._num_epochs_run = self._num_epochs_run + num_epochs
    # def run_epochs_till_convergence(self, X_train, y_train, X_val=None, y_val=None):
    #     epoch = 0
    #     last_val_err = 10
    #     num_bad_vals = 0
    #     while True:
    #         self._start_time = time.time()

    #         train_err = self._run_an_epoch(X_train, y_train, self._train_fn)
    #         self.cost_history['train_costs'].append(train_err)

    #         if self.verbose: 
    #             print_progress(epoch, 0, train_err, start_time,
    #                                 val_err=val_err, 
    #                                 final_epoch=False, 
    #                                 print_period=100)


    #         if X_val is not None and y_val is not None:
    #             val_err = self._run_an_epoch(X_val, y_val, self._val_fn)
    #             self.cost_history['val_costs'].append(val_err)
    #             if self.verbose: 
    #                 print_progress(epoch, num_epochs, train_err,
    #                                     val_err=val_err, 
    #                                     final_epoch=False, 
    #                                     print_period=round(num_epochs/10))
               
    #         if epoch > 5000 and val_err > last_val_err:
    #             num_bad_vals += 1
    #         last_val_err = val_err
            
    #         if num_bad_vals > 100:
    #             print_progress(epoch, num_epochs, train_err,
    #                                 val_err=val_err,
    #                                 final_epoch=True,
    #                                 print_period=1)

    #             break
            
    #         epoch += 1
            # self._num_epochs_run += 1

    def _run_an_epoch(self, X, y, cost_fn):
        # print(len(y))
        num_examples = len(y)
        # print(num_examples)
        current_err = 0
        batches = 0
        batch_perc = DEFAULT_BATCH_PERC

        if self.batch_size is not None:
            if num_examples<(5*self.batch_size):
                batch_size = int(batch_perc*num_examples)
            else:
                batch_size = self.batch_size
        else:
            batch_size = int(batch_perc*num_examples)
        # print(batch_size)
        for batch in self._iterate_minibatches(X, y, 
                                               batch_size, 
                                               shuffle=True):
            inputs, targets = batch
            
            current_err += cost_fn(inputs, targets)
            batches += 1
        # print(batches)
        current_err /= batches
        # current_err /=num_examples
        self._num_epochs_run += 1
        return current_err


    def _iterate_minibatches(self,inputs, targets, batchsize, shuffle=True):
        assert(len(inputs) == len(targets))
        
        # if not self.is_recurrent:
            # num_time_steps = inputs.shape[1]

            # assert(num_time_steps == targets.shape[1])
        # num_in_feats = inputs.shape[1]
        # num_out_feats = targets.shape[1]
            # inputs = np.reshape(inputs, (-1, num_in_feats))
            # targets = np.reshape(targets, (-1, num_out_feats))
            # batchsize *= num_time_steps
        
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):

            # print(batchsize)
            # print(inputs.shape)
            # print(len(inputs))
            # print(start_idx)

            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize,1)
            # print(excerpt)
            if inputs.ndim==2:
                yield inputs[excerpt,:], targets[excerpt,:]
            elif inputs.ndim==3:
                yield inputs[excerpt,:,:], targets[excerpt,:,:]
            elif inputs.ndim==4:
                yield inputs[excerpt,:,:,:], targets[excerpt,:,:,:]
            elif inputs.ndim==5:
                yield inputs[excerpt,:,:,:,:], targets[excerpt,:,:,:,:]

    # def _run_an_epoch(self,X, y, cost_fn):
    #     num_examples=len(y)
    #     # print(num_examples)
    #     current_err = 0
    #     batches = 0
    #     batch_perc = DEFAULT_BATCH_PERC
    #     for batch in self._iterate_minibatches(X, y, 
    #                                      int(batch_perc*num_examples), 
    #                                      shuffle=False):

    #         inputs, targets = batch


    #         # self.batch_means.append(np.mean(targets))
    #         # self.batch_vars.append(np.var(targets))
    #         # self.batch_sizes.append(np.shape(targets))

    #         current_err += cost_fn(inputs, targets)
    #         batches += 1
    #     current_err /= batches
    #     # current_err /=num_examples
    #     return current_err

    #-------------------------------------------------------------------------------
    #-----------------------------Public functions----------------------------------
    #-------------------------------------------------------------------------------
    def optimize(self, X_train, y_train, X_val=None, y_val=None, num_epochs = 100):
        self._orig_start_time = time.time()
        self.run_n_epochs(X_train, y_train, X_val, y_val, num_epochs)
        self.cost_history['final_train_cost'] = self.cost_history['train_costs'][-1]
        if self.cost_history['val_costs']:
            self.cost_history['final_val_cost'] = self.cost_history['val_costs'][-1]

        network_params = lasagne.layers.get_all_param_values(self.predictive_network.network)
        return network_params, self.cost_history


#-------------------------------------------------------------------------------
#-----------------------------Output functions----------------------------------
#-------------------------------------------------------------------------------
#This is a function rather than a class method bc it does not rely on any
#class attributes
def print_progress(epoch, num_epochs, train_err, start_time,
                   val_err=None, 
                   final_epoch=False, 
                   print_period=10):
    min_period = 10
    if epoch < min_period or epoch % print_period == 0:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training cost:\t\t{:.6f}".format(train_err))
        if val_err is not None:
            print("  validation cost:\t\t{:.6f}".format(val_err))

    if final_epoch:
        print("Total time took {:.3f}s".format(time.time() - start_time))
    return



