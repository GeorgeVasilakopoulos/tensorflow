import os

os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'

import tensorflow as tf


cluster_spec = {
        "local": ["localhost:2222", "localhost:2223"]
    }

server = tf.distribute.Server(cluster_spec, job_name="local", task_index=0)
server.join()