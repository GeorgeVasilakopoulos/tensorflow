import tensorflow as tf
from tensorflow.python.framework import function

# fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, func_name="Fac", out_names=["ret"])
def FacImpl(n):
	return tf.cond(tf.less_equal(n, 1),
		lambda: tf.constant(1),
		lambda: n * FacImpl(n - 1))


print(FacImpl(5))


