# Distributed Grid Search

Code to perform distributed grid search, using multiple nodes.

Flexibly choose hyperparameters to be run. 
These Cartesian product of these parameters is queued (using RabbitMQ) on a central host server. 
The queue can be polled by any number of processes on the host machine or on remote machines. 
