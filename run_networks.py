import sys
import os
from imp import reload
from collections import OrderedDict
import numpy as np
import network_runner
import task_scheduler
reload(network_runner)
reload(task_scheduler)

#Set the Theano variables, selecting the desired GPU
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32,"

DEFAULT_TASK_NAME = 'my_grid_search'
DEFAULT_HOSTNAME = '123.1.234.56' #The address of the server where you will host the message queue
DEFAULT_USERNAME = 'my_username' #Your username on the host machine
DEFAULT_DATA_PATH = '/full/path/to/data/'
DEFAULT_MAIN_SAVE_PATH = '/full/save/folder/path/' #If this doesn't exist, it will be created

def set_append_keys():
    """
    By default, will only include variable names in filename if there are multiple values
    for it. Can manually override this for specific hyperparameters by listing them here.
    """
    append_keys = ['num_filters', 'nonlinearity']
    return append_keys

def set_hyperparameters(data_path, main_save_path):
    '''
    Define the set of hyperparameters to be run.
    You can define as many or as few hyperparameters as you would like.
    For each hyperparameter, you can set one or more values to be tried.
    The Cartesian prodict of all of the hyperparameters will be explored.
    '''
    hyperparameters = OrderedDict()

    hyperparameters['data_path'] = data_path
    hyperparameters['main_save_path'] = main_save_path

    hyperparameters['num_filters'] = np.array([200, 400, 800])
    hyperparameters['regularization'] = ['l1']
    hyperparameters['reg_factor'] = np.logspace(-5, -7, 5)
    hyperparameters['nonlinearity'] = ['relu']
    hyperparameters['output_nonlinearity'] = [None]
    hyperparameters['num_layers'] = [1]
    hyperparameters['update_func'] = ['adam']
    hyperparameters['num_epochs'] = [1000]
    hyperparameters['noise_ratio'] = [0, 0.5]
    hyperparameters['norm_type'] = [1]

    return hyperparameters

def main(option, task_name=DEFAULT_TASK_NAME):
    '''
    @param: option
    valid options are: schedule, receive, or clear
    '''
    # task_name = DEFAULT_TASK_NAME
    main_hostname = DEFAULT_HOSTNAME
    username = DEFAULT_USERNAME
    data_path = DEFAULT_DATA_PATH
    main_save_path = DEFAULT_MAIN_SAVE_PATH

    hyperparameters = set_hyperparameters(data_path, main_save_path)
    append_keys = set_append_keys()

    if option == 'schedule':
        print('Scheduling tasks...')
        task_scheduler.schedule_tasks(hyperparameters, task_name, main_hostname,
                                      add_save_paths=True,
                                      main_save_path=main_save_path,
                                      append_keys=append_keys,
                                      shuffle=True)
    elif option == 'receive':
        print('Receiving tasks...')
        nr = network_runner.NetworkRunner(task_name, main_hostname,
                                          username=username,
                                          start_listening_on_init=True)
    elif option == 'clear':
        print('Clearing scheduler...')
        task_scheduler.clear_tasks(task_name, main_hostname)
    # elif option == 'launch_receivers':
    #     print('launching all set receicers in parallel...')
    #     launch_receivers()
    else:
        raise('Invalid option! Please select: schedule, receive or clear.')

if __name__ == "__main__":
    main(*sys.argv[1:])
