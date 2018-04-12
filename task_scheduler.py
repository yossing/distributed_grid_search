import socket
import random
import itertools
from imp import reload
import pandas as pd
import numpy as np
import message_handler
reload(message_handler)


def clear_tasks(task_name, main_hostname):
    '''
    Clears scheduler by receiveing tasks and printing them rather than 
    using them to run networks
    '''
    this_hostname = socket.gethostname()
    if this_hostname in main_hostname:
        host = 'localhost'
    else:
        host = main_hostname
    parameter_scheduler = message_handler.MessageHandler(task_name, host)
    parameter_scheduler.receive_messages(print)

def schedule_tasks(dict_of_hyperparameter_lists, task_name, main_hostname,
                   add_save_paths=True, main_save_path='', append_keys=[],
                   shuffle=False):

    hyperparameters_as_json_list = format_hyperparameters(dict_of_hyperparameter_lists,
                                                          add_save_paths=add_save_paths,
                                                          main_save_path=main_save_path,
                                                          append_keys=append_keys)

    parameter_scheduler = message_handler.MessageHandler(task_name, host=main_hostname)
    print(len(hyperparameters_as_json_list))
    #randomize the order of hyperparameter search. Gives quicker intuition of partially run results
    if shuffle:
        random.shuffle(hyperparameters_as_json_list)
    parameter_scheduler.send_messages(hyperparameters_as_json_list)


def format_hyperparameters(dict_of_hyperparameter_lists,
                           add_save_paths=True,
                           main_save_path='',
                           append_keys=[]):

    '''
    Format hyperparameters into usable list and schedule these with a message_handler
    '''
    hyperparameters_as_list = explode_to_list(dict_of_hyperparameter_lists)
    hyperparameters_df = pd.DataFrame(hyperparameters_as_list)
    #Add save_folder path to hyperparameter df
    multi_value_keys = []
    selected_keys = multi_value_keys

    if add_save_paths:
        for key, value in sorted(dict_of_hyperparameter_lists.items()):
            # n_items = len([item for item in value if item])
            n_items = len(value)
            if n_items > 1:
                multi_value_keys.append(key)
        selected_keys = multi_value_keys
        print('multi value keys: ', selected_keys)
        print('specified keys to append: ', append_keys)

        for key in append_keys:
            if key not in selected_keys:
                selected_keys.append(key)

        hyperparameters_df = add_folder_names(hyperparameters_df, selected_keys,
                                              main_save_path=main_save_path)
    #convert each entry into a json string and add to list.
    hyperparameters_as_json_list = []
    for row_ix in range(len(hyperparameters_df)):
        hyperparameters_as_json_list.append(hyperparameters_df.iloc[row_ix].to_json())

    return hyperparameters_as_json_list


def explode_to_list(d):
    '''
    Convert a dictionary of lists into a list of dictionaries of every possible combination
    '''
    exploded_list = []
    try:
        for vals in itertools.product(*d.values()):
            exploded_list.append(dict(zip(d.keys(), vals)))
    #This will fail when all of the dict entries have only one element.
    #Handle this case with a catch:
    except TypeError:
        exploded_list.append(d)
    return exploded_list

def add_folder_names(df, selected_keys, main_save_path='', row_name='save_path'):
    '''
    add a column to the input df called 'folder_name'. Going through selected keys
    '''
    def name_this_folder(row):
        '''
        row-wise function to be applied to df.
        loop through selected keys and append key and value to string
        '''
        name = ''
        for key in selected_keys:
            val = row[key]
            if not isinstance(val, str):
                val = np.around(val, 10)
            if name == '':
                name = main_save_path
            else:
                name = name + '_'
            name = name + key + '_' + str(val)
        return name
    df[row_name] = df.apply(name_this_folder, axis=1)
    return df
