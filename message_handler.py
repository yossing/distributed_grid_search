"""
Module to send and receive messages on a queue using pika, the Python interface for RabbitMQ. 
Author: Yosef Singer
"""
import pika
import json
import traceback

DEFAULT_HEARTBEAT_INTERVAL = 60 #Number of seconds after which connection will timeout if there is no comm
DEFAULT_HOSTNAME = '123.1.234.56' #The address of the server where you will host the message queue
DEFAULT_RABBITMQ_USERNAME = 'my_rabbitmq_username' 
DEFAULT_RABBITMQ_PASSWORD = 'my_rabbitmq_password'

class MessageHandler(object):
    connection = None
    task_name = None

    def __init__(self, task_name, host=DEFAULT_HOSTNAME):
        self.task_name = task_name
        self.host = host
        self.user = DEFAULT_RABBITMQ_USERNAME
        self.password = DEFAULT_RABBITMQ_PASSWORD
        self.virtual_host = 'my_virtual_host'
        self.connect()

    def connect(self):

    	#These are your RabbitMQ credentials and will allow you to open a connection on the machine. 
    	#Note that these are not your login credentials to the host machine. 
    	#This assumes you have setup RabbitMQ securely to allow only registered users to open connections. 
        credentials = pika.PlainCredentials(self.user, self.password) 

        parameters = pika.ConnectionParameters(host=self.host,
                                               port=5672,
                                               virtual_host=self.virtual_host,
                                               credentials=credentials, 
                                               heartbeat_interval=DEFAULT_HEARTBEAT_INTERVAL)
        self.connection = pika.BlockingConnection(parameters=parameters)
        print("Connection established to task scheduler on %s."%self.host)

    def send_messages(self, message_list):
        '''
        Method to define provider, setup queue and send messages
        '''
        if self.connection.is_closed:
            self.connect()
        channel = self.connection.channel()
        channel.queue_declare(queue=self.task_name, durable=True)

        # self.messages_pending = True
        for message in message_list:
            channel.basic_publish(exchange='',
                                  routing_key=self.task_name,
                                  body=message,
                                  properties=pika.BasicProperties(
                                      delivery_mode=2, # make message persistent
                                  ))
            print(" [x] Sent %r" % message)
    	# self.messages_pending = False
        self.connection.close()
        return


    def receive_messages(self, function_handle, quit_on_error=True):
        '''
        Method which receives messages from existing queue and passes message to
        function_handle on receipt

        '''        
        def callback(ch, method, properties, message_body):
            try:
                '''
                callback method to handle incoming messages to consumer
                '''
                # process message
                print(" [x] Received message")
                #Let's acknowledge receiving the message straight away. 
                #If there is an exception, it will manually be readded to the queue.
                ch.basic_ack(delivery_tag=method.delivery_tag)

                #This assumes you have formatted your messages as JSON strings
                message_body = message_body.decode()
                message_body = json.loads(message_body)

                #Close the connection, as we don't need it open while processing the message.
                self.connection.close()
                #Now handle the message
                function_handle(message_body)
                print(" [x] Done handling message")

            except Exception as e:#pika.exceptions.ConnectionClosed:
                print('*** Caught exception: ' + str(e.__class__) + ': ' + str(e))
                traceback.print_exc()

                # print('oops. lost connection. trying to reconnect.')
                print('Failed to complete handling message. Re-adding to the queue.')
                self.send_messages([message_body])
                if quit_on_error:
                    raise(e)
                #start receiving messages again
                # self.receive_messages(function_handle)    

        while True:

            if self.connection.is_closed:
                self.connect()
            channel = self.connection.channel()
            #We assume that all of the messages have been sent before we start receiving. 
            #If there are no messages in the queue, exit. 
            qd = channel.queue_declare(queue=self.task_name, durable=True)
            if qd.method.message_count==0:
                print("There are no messages to receive. Closing connection and exiting.")
                self.connection.close()
                break

            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(callback,
                                  queue=self.task_name)
            channel.start_consuming()


