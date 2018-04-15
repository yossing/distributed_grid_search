# Distributed Grid Search

Python code to perform distributed grid search using one or more nodes.

Flexibly choose hyperparameters to be run. 
The Cartesian product of these parameters is queued (using RabbitMQ) on a central host server. 
The queue can be polled by any number of processes on the host or remote machines, allowing for massive embarrasingly parallel computing. 

All training data and results are stored on the host machine and are copied automatically to each remote machine as needed. When a job is run on a remote machine, the result is saved locally and copied to the host so that all of the results are centrally stored. This eliminates the need to rely on external products like Dropbox to sync data between machines. 

The code is not dependant on any framework (it can easily be used with Theano or Tensorflow). 

# Installation instructions
1. Install RabbitMQ onto the host machine: https://www.rabbitmq.com/install-debian.html
2. Setup RabbitMQ securely on the host machine, following the instructions [here](./RabbitMQ_secure_setup.md).
3. Create a virtual environment with the following python packages: 
  - pika
  - numpy
  - scipy
  - paramiko
  - scp
  optionally, if you would like to try the examples: 
  - theano
  - lasagne
4. Clone the git repository 
