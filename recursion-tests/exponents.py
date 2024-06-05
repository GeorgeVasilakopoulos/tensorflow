import os
import tensorflow as tf
from tensorflow.python.framework import function

os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()

exp = function.Declare("EXPONENT", [("x", tf.float32), ("n", tf.int32)], [("ret", tf.float32)])




# fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.float32, tf.int32, func_name="EXPONENT", out_names=["ret"])
def ExpImpl(x, n):
    return tf.cond(tf.equal(n,0),
                lambda: tf.cast(tf.constant(1),tf.float32),
                lambda: x*x)


# @function.Defun(tf.int32, func_name="Fac", out_names=["ret"])
# def FacImpl2(n):
# 	return t(1)


ExpImpl.add_to_graph(tf.compat.v1.get_default_graph())
# t.add_to_graph(tf.compat.v1.get_default_graph())
# FacImpl2.add_to_graph(tf.compat.v1.get_default_graph())


x = tf.compat.v1.get_variable('n_var', [], initializer=tf.constant_initializer(4.0))
y = ExpImpl(x,2)

train_op = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(y)
print(tf.compat.v1.get_default_graph().as_graph_def())


sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.initialize_all_variables())
print(x.eval(session=sess))
print(sess.run(train_op))
print(x.eval(session=sess))

# writer = tf.summary.FileWriter('./graphs', tf.compat.v1.get_default_graph())

# with tf.compat.v1.Session() as sess:
#     result = ExpImpl(2,5)
#     print("Result:", sess.run(result))

