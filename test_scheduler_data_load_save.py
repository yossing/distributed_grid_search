import sys
import os
# import ntpath
import numpy as np
import pickle as pkl
from imp import reload
from utils import tic,toc
from collections import OrderedDict
from task_runner import TaskRunner
import task_scheduler 
reload(task_scheduler)

TASK_NAME = 'test_copying'
MAIN_HOSTNAME = '123.1.234.56' #The address of the server where you will host the message queue
DEFAULT_USERNAME = 'my_username' 

main_load_path = '~/Desktop/test_copying/level_1/branch_2/level_2'
main_save_path = '~/Desktop/test_copying/level_1/branch_3/'

def schedule_task():
    params = OrderedDict()
    params['main_load_path'] = [main_load_path]
    params['main_save_path'] = [main_save_path]
    params['file_num'] = np.arange(100,101)
    task_scheduler.schedule_tasks(params, TASK_NAME, MAIN_HOSTNAME, add_save_paths=False)
    return
def run_task(which_task):
    if which_task == 'clear':
        task_scheduler.clear_tasks(TASK_NAME, MAIN_HOSTNAME)
    if which_task =='schedule':
        schedule_task()
    if which_task=='save':
        my_runner = DataSaver(TASK_NAME, MAIN_HOSTNAME, username = DEFAULT_USERNAME, start_listening_on_init=True)
    elif which_task=='load':
        my_runner = DataLoader(TASK_NAME, MAIN_HOSTNAME, username = DEFAULT_USERNAME, start_listening_on_init=True)
    elif which_task=='load_and_save':
        my_runner = DataLoaderAndSaver(TASK_NAME, MAIN_HOSTNAME, username = DEFAULT_USERNAME, start_listening_on_init=True)
    return

class DataSaver(TaskRunner):
    def run(self, params):
        print('received the following params: ')
        print(params)
        def pickle_data(save_path, data, protocol=4):
            pkl.dump(data, open(save_path, 'wb'), protocol=protocol)
            return
        self.save_data(params['main_save_path'], 'file_%i.pkl' %(params['file_num']), pickle_data, params)
        return

class DataLoader(TaskRunner):
    def run(self, params):
        print('received the following params: ')
        print(params)
        def unpickle_data(pth):
            return pkl.load(open(pth, 'rb'))
        file_path = os.path.join(params['main_load_path'], 'file_%i.pkl' %params['file_num'])
        print('trying to load: %s' %file_path)
        try:
            data = self.load_data(file_path, unpickle_data)
            print('successfully loaded: ')
            print(data)
        except FileNotFoundError:
            print('Could not load the specified file, it was not present on remote or local machine!')
        return

class DataLoaderAndSaver(TaskRunner):
    def run(self, params):
        print('received the following params: ')
        print(params)
        def unpickle_data(pth):
            return pkl.load(open(pth, 'rb'))
        file_path = os.path.join(params['main_load_path'], 'file_%i.pkl' %params['file_num'])
        print(file_path)
        data = self.load_data(file_path, unpickle_data)
        print('successfully loaded: ')
        print(data)

        def pickle_data(save_path, data, protocol=4):
            pkl.dump(data, open(save_path, 'wb'), protocol=protocol)
            return
        self.save_data(params['main_save_path'], 'file_%i.pkl' %params['file_num'], pickle_data, params)
        return

if __name__ == "__main__":
    run_task(sys.argv[1])


