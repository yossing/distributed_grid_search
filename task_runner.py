from imp import reload
import sys
import os
import random
import numpy as np
import scipy.io
import socket
import ssh
import ntpath
from collections import OrderedDict

import message_handler
from utils import tic, toc
reload(ssh)
reload(message_handler)

class TaskRunner(object):
    task_name = None
    main_hostname = None
    username = None

    def __init__(self, task_name, main_hostname, data_path=None, data_loading_fn = None, username = 'yossi', start_listening_on_init=True):
        print('Initializing Task Runnner...')
        self.task_name = task_name
        this_hostname = socket.gethostname()
        if this_hostname.lower() not in main_hostname.lower():
            self.main_hostname = main_hostname
        else:
            self.main_hostname = 'localhost'
        self.username = username
        
        if start_listening_on_init:
            self.start_message_listener()
        return

    def start_message_listener(self):
        #Start listening for messages queue
        my_msg_handler = message_handler.MessageHandler(self.task_name, host=self.main_hostname)
        my_msg_handler.receive_messages(self.run)
        return

    def run(self, *args, **kwargs):
        """
        This is an abstract method that must be implemented in the child class
        """
        raise NotImplementedError()

    def load_data(self, file_path, data_loading_fn, *args, **kwargs):
        # print(os.path.join(data_path,filename))
        local_file_path = os.path.expanduser(file_path) 
        self.file_path = local_file_path
        # print(os.path.isfile(os.path.join(local_file_path,filename)))
        print('loading file: %s' %file_path)
        if not os.path.isfile(local_file_path) and self.main_hostname != 'localhost':
            print('Copying data from remote machine...')
            tic()
            ssh.copy_file_from_remote(file_path, self.main_hostname, self.username)
            print('Completed copying data from main host.')
            toc()
        print('Loading data...')
        tic()
        data = data_loading_fn(local_file_path, *args, **kwargs)
        toc()
        return data
    
    def save_data(self, save_path, save_fn, *args, **kwargs):
        #first save the file locally
        save_fn(*args, **kwargs)
        #now copy to remote if we are not on the main host machine
        if self.main_hostname != 'localhost':
            ssh.copy_data_to_remote(save_path, self.main_hostname, self.username)
        return 

