import os


os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.framework import function

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()


cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])





@function.Defun(tf.int32, func_name="Fac", out_names=["ret"])
def FacImpl(n):

	def f1(): 
		with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
			ret = tf.constant(1)
		return ret
	def f2(): 
		with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
			ret = n * fac(n - 1)
		return ret

	with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
		pred = tf.less_equal(n, 1)

	return tf.cond(pred, f1, f2)

FacImpl.add_to_graph(tf.compat.v1.get_default_graph())

n = tf.constant(10)
x = fac(n)

#print(tf.get_default_graph().as_graph_def())

# writer = tf.compat.v1.summary.FileWriter('./graphs', tf.compat.v1.get_default_graph())

with tf.compat.v1.Session("grpc://localhost:2222") as sess:
	print(sess.run(x))

# writer.close()
