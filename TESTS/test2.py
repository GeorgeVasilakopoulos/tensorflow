import os

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'
os.environ['TF_DUMP_GRAPH_NAME_FILTER'] = 'Fac'
import tensorflow as tf
from tensorflow.python.framework import function



tf.compat.v1.disable_eager_execution()

# tf.logging.set_verbosity(tf.logging.INFO)
fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])


@function.Defun(tf.int32, func_name="Test", out_names=["ret"])
def t(n):
	return tf.constant(1)



# fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, func_name="Fac", out_names=["ret"])
def FacImpl(n):
	return tf.cond(tf.less_equal(n, 1),
		lambda: tf.constant(1),
		lambda: n * fac(n - 1))


# @function.Defun(tf.int32, func_name="Fac", out_names=["ret"])
# def FacImpl2(n):
# 	return t(1)


FacImpl.add_to_graph(tf.compat.v1.get_default_graph())
# t.add_to_graph(tf.compat.v1.get_default_graph())
# FacImpl2.add_to_graph(tf.compat.v1.get_default_graph())


print(tf.compat.v1.get_default_graph().as_graph_def())


# writer = tf.compat.v1.summary.FileWriter('/tensorflow/TESTS/graph', tf.compat.v1.get_default_graph())
# writer = tf.summary.FileWriter('./graphs', tf.compat.v1.get_default_graph())

with tf.compat.v1.Session() as sess:
    result = FacImpl(1)
    print("Result:", sess.run(result))

