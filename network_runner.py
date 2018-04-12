from imp import reload
import sys
import os
import ntpath
import numpy as np
import data_handling as dat
import predictive_network_subclasses as pn
import utils
import message_handler
from task_runner import TaskRunner
reload(utils)
reload(message_handler)
reload(dat)
reload(pn)


class NetworkRunner(TaskRunner):
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    data_loading_settings = {'noise_ratio':0,
                             'norm_type':1}
    data_path = None

    def __init__(self, *args, **kwargs):
        self.data_loading_fn = dat.load_1d_conv_vis_data
        self.pn_subclass = pn.PredictiveConv1DNetwork
        super().__init__(*args, **kwargs)

    def run(self, network_settings):
        '''
        Main network running method. Can use independantly or pass into 
        message_handler as a callback method.
        @param: network_settings:
         - A dict of kwargs with specific settings for this network
        '''
        print('Received network settings: ')
        print(network_settings)

        save_name = 'python_network.pkl'
        save_path = network_settings['save_path']
        if save_path.endswith(save_name):
            full_save_path = save_path
        else:
            full_save_path = os.path.join(save_path, save_name)

        network_settings['save_path'] = full_save_path
        
        #Let's make sure we are dealing with the same data_path as originally set to deal with. 
        data_path = network_settings['data_path']
        #update data_loading_settings with any that might overwrite defaults
        prev_data_loading_settings = self.data_loading_settings.copy()
        for key, value in network_settings.items():
            if key in self.data_loading_settings:
                self.data_loading_settings[key] = value
        #Check if any settings affecting the training dataset has changed. If so, reload it with the new settings.
        if self.data_path != os.path.expanduser(data_path) or not all(self.data_loading_settings == prev_data_loading_settings):
        #If not, then set data path to new path and load data. 
            filedir, filename = ntpath.split(data_path)
            print(filedir)
            print(filename)
            if filename == '':
                filename = 'normalized_concattrain.pkl'

            self.X_train, self.y_train, self.X_val, self.y_val = self.load_data(os.path.join(filedir, filename), 
                                                                                self.data_loading_fn, 
                                                                                **self.data_loading_settings)
    
        in_shape = self.X_train.shape
        out_shape = self.y_train.shape
        current_net = self.pn_subclass(in_shape, out_shape, **network_settings) 
        
        print('Training network...')
        num_epochs = network_settings['num_epochs']
        n_iter = 10
        for ii in range(n_iter):
            current_net.train_network(self.X_train, self.y_train,       
                                      X_val=self.X_val, y_val=self.y_val,
                                      num_epochs=int(num_epochs//n_iter), 
                                      show_graph=False,
                                      max_epochs=int(num_epochs))
            print('Completed %i percent of network training' %int(100*(ii+1)//10))
            print('saving...')
            current_net.save_path = full_save_path
            self.save_data(full_save_path, current_net.to_pickle)
        return
