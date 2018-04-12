import traceback
import os
import ntpath
import paramiko
from scp import SCPClient
from utils import tic,toc
# import sys

def start_ssh(username, server, verbose=True):
    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(server, username=username, allow_agent=True)
    if verbose:
        ssh_stdin, ssh_stdout, ssh_stderr = ssh_client.exec_command('hostname')
        print('Connected to :')
        print_output(ssh_stdout)
    return ssh_client

def print_output(ssh_stdout):
        for line in ssh_stdout:
            print(line.strip('\n'))

def ssh_cmd(username, server, cmd_to_execute, verbose=True, show_output = True):
    try:
        ssh_client=start_ssh(username, server, verbose=verbose)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh_client.exec_command(cmd_to_execute)
    # try:
    #     ssh_client.close()
    # except:
    #     pass
    # sys.exit(1)
        output_text = []
        for line in ssh_stdout:
            output_text.append(line.strip('\n'))
        if show_output:
            if output_text:
                print(output_text)
        return output_text
    except Exception as e:
        print('*** Caught exception: ' + str(e.__class__) + ': ' + str(e))
        traceback.print_exc()
        # return e
    
    # return ssh_stdin, ssh_stdout, ssh_stderr


def scp_put(username, server, local_path, remote_path, recursive=False, verbose = True):
    ssh_client = start_ssh(username, server, verbose=verbose)
    scp = SCPClient(ssh_client.get_transport())
    scp.put(local_path, remote_path, recursive=recursive)
    scp.close()

def scp_get(username, server, remote_path, local_path, recursive=False, verbose = True):
    ssh_client=start_ssh(username, server, verbose=verbose)
    scp = SCPClient(ssh_client.get_transport())
    scp.get(remote_path, local_path, recursive=recursive)
    scp.close()

    
def copy_file_from_remote(file_path, main_hostname, username):
    """
    Copy file or folder from remote to local machine using scp. 
    This uses Paramiko's scp module and assumes that ssh keys between the two machines have been setup.
    @file_path: This should lead with the generic '~/' user sign. This makes the home path compatible
                between different machines
    @main_hostname: The name or ... of the remote machine
    @username: The username used for login to remote machine
    """
    raw_file_path = file_path
    local_file_path = os.path.expanduser(file_path)
    # if not os.path.isfile(local_file_path+filename) and main_hostname != 'localhost':
    local_parent_dir, filename = ntpath.split(local_file_path)
    if not os.path.isdir(local_parent_dir):
        print('Path to parent directory %s does not exist, creating dir.' %local_parent_dir)
        os.makedirs(local_parent_dir)
    print('Copying data from %s via scp...' %main_hostname)
    tic()
    # copy the data folder (and contents) from the remote machine to the parent directory on the local machine
    # parent_dir = os.path.dirname(os.path.dirname(self.local_file_path))
    # local_parent_dir = os.path.dirname(local_file_path)
    # ssh.scp_get(username, main_hostname, raw_file_path, os.path.join(file_path, os.pardir), recursive=True)
    scp_get(username, main_hostname, raw_file_path, local_parent_dir, recursive=True)
    print('Completed copying data from main host.')
    toc()
    return

def copy_data_from_remote(data_path, main_hostname, username):
    """
    Copy file or folder from remote to local machine using scp. 
    This uses Paramiko's scp module and assumes that ssh keys between the two machines have been setup.
    @data_path: This should lead with the generic '~/' user sign. This makes the home path compatible
                between different machines
    @main_hostname: The name or ... of the remote machine
    @username: The username used for login to remote machine
    """
    raw_data_path = data_path
    local_data_path = os.path.expanduser(data_path)
    # if not os.path.isfile(local_data_path+filename) and main_hostname != 'localhost':
    if not os.path.isdir(local_data_path):
        print('Path to directory %s does not exist, creating dir.' %local_data_path)
        os.makedirs(local_data_path)
    print('Copying data from main host via scp...')
    tic()
    # copy the data folder (and contents) from the remote machine to the parent directory on the local machine
    # parent_dir = os.path.dirname(os.path.dirname(self.local_data_path))
    local_parent_dir = os.path.dirname(local_data_path)
    # scp_get(username, main_hostname, raw_data_path, os.path.join(data_path, os.pardir), recursive=True)
    scp_get(username, main_hostname, raw_data_path, local_parent_dir, recursive=True)
    print('Completed copying data from main host.')
    toc()
    return

def copy_data_to_remote(save_path, main_hostname, username):
    """
    Copy file or folder from local to machine to remote using scp. 
    This uses Paramiko's scp module and assumes that ssh keys between the two machines have been setup.
    @save_path: This should lead with the generic '~/' user sign. This makes the home path compatible
                between different machines
    @main_hostname: The name or ... of the remote machine
    @username: The username used for login to remote machine
    """
    if main_hostname != 'localhost':
        print('Copying results to: %s...' %main_hostname)
        #copy the saved folder (and contents) to the parent directory on the remote machine
        remote_parent_dir = os.path.dirname(save_path)
        local_save_path = os.path.expanduser(save_path)
        cmd_outtext = ssh_cmd(username, main_hostname, 'mkdir -p %s' %remote_parent_dir)
        if cmd_outtext:
            print(cmd_outtext)
        scp_put(username, main_hostname, local_save_path, remote_parent_dir, recursive=True)
    return 


# def launch_task_listeners(hosts):
#     import threading
#     # def launch_listener_host():
#     thrs = []
#     for host in hosts:
#         print('starting new thread for: %s' %host)
#         thr = threading.Thread(target=test_ssh, args =('yossi', host), kwargs={})
#         thr.start() # will run "foo"
#         thrs.append(thr)
#         thr.is_alive() # will return whether foo is running currently
#     for thr in thrs:
#         thr.join() # will wait till "foo" is done

# key = paramiko.RSAKey(data=base64.b64decode(b'lin-ajk005.dpag.ox.ac.uk'))
# client = paramiko.SSHClient()
# client.get_host_keys().add('ssh.example.com', 'ssh-rsa', key)
# client.connect('ssh.example.com', username='strongbad', password='thecheat')
# stdin, stdout, stderr = client.exec_command('ls')
# for line in stdout:
#     print('... ' + line.strip('\n'))
# client.close()