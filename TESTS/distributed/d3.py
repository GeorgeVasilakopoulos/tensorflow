import os


os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'

import tensorflow as tf


cluster_spec = {
        "local": ["172.19.0.3:2222", "172.19.0.2:2223"]
    }

server = tf.distribute.Server(cluster_spec, job_name="local", task_index=1)
server.join()
