import tensorflow as tf
from tensorflow.python.framework import function

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()

tak = function.Declare("Tak", [("x", tf.int32), ("y", tf.int32), ("z", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, tf.int32, tf.int32, func_name="Tak", out_names=["ret"])
def TakImpl(x,y,z):
	return tf.cond(tf.less(y, x),
        lambda: tak(tak(x-1,y,z), tak(y-1,z,x), tak(z-1,x,y)),
		lambda: z)

TakImpl.add_to_graph(tf.compat.v1.get_default_graph())


with tf.compat.v1.Session() as sess:
    result = TakImpl(24,16,8)
    print("Result:", sess.run(result))

#print(tf.get_default_graph().as_graph_def())