#!/usr/bin/env python
import pika
import time
import threading
connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost',
        heartbeat_interval=20))
channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True)
print ' [*] Waiting for messages. To exit press CTRL+C'
def thread_func(ch, method, body):
    long_running_task(connection)
    ch.basic_ack(delivery_tag = method.delivery_tag)
def callback(ch, method, properties, body):
    threading.Thread(target=thread_func, args=(ch, method, body)).start()
channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
                      queue='task_queue')
channel.start_consuming()