import tensorflow as tf


cluster_spec = {
        "worker": ["localhost:2222", "localhost:2223"]
    }

server = tf.distribute.Server(cluster_spec, job_name="worker", task_index=1)
server.join()