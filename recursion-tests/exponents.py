import os
import tensorflow as tf
from tensorflow.python.framework import function



tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()

exp = function.Declare("EXPONENT", [("x", tf.int32), ("n", tf.int32)], [("ret", tf.int32)])




# fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, tf.int32, func_name="EXPONENT", out_names=["ret"])
def ExpImpl(x, n):
    return tf.cond(tf.equal(n,0),
                lambda: tf.constant(1),
                lambda: x*exp(x,n-1))

# @function.Defun(tf.int32, func_name="Fac", out_names=["ret"])
# def FacImpl2(n):
# 	return t(1)


ExpImpl.add_to_graph(tf.compat.v1.get_default_graph())
# t.add_to_graph(tf.compat.v1.get_default_graph())
# FacImpl2.add_to_graph(tf.compat.v1.get_default_graph())

# writer = tf.summary.FileWriter('./graphs', tf.compat.v1.get_default_graph())

with tf.compat.v1.Session() as sess:
    result = ExpImpl(2,5)
    print("Result:", sess.run(result))

