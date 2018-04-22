All necessary info can be found here: https://www.rabbitmq.com/man/rabbitmqctl.1.man.html

First delete the guest user that comes as a default:
 sudo rabbitmqctl delete_user guest
Next install new secure user with a good password. eg:
 sudo rabbitmqctl add_user '''user psswd'''

Add a virtual host which you will use to schedule your processes:
 sudo rabbitmqctl add_vhost '''my_host'''

Give the secure user permissions on the virtual host:
 sudo rabbitmqctl set_permissions -p '''my_host user''' ".*" ".*" ".*"

Now, to establish a connection using pika, you would use something like:
 credentials = pika.PlainCredentials('''user''', '''psswd''')
 parameters = pika.ConnectionParameters('''host_machine''',
                                        5672,
                                        '''my_host''',
                                        credentials)
 self.connection = pika.BlockingConnection(parameters=parameters)

as is detailed here: https://pika.readthedocs.io/en/latest/modules/parameters.html
