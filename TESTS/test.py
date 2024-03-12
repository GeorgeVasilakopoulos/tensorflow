import tensorflow as tf
from tensorflow.python.framework import function

log_dir = "./graph"

tf.config.run_functions_eagerly(False)

fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])
writer = tf.summary.create_file_writer(log_dir)


def FacImpl(n):
    condition = tf.equal(n, tf.constant(1))
    true_branch = tf.constant(1)
    subtraction_op = tf.subtract(n, 1)
    false_branch = tf.multiply(fac(subtraction_op), n)
    return tf.cond(condition, lambda: true_branch, lambda: false_branch)

transformed_function = tf.function(FacImpl)

# print(tf.get_default_graph().as_graph_def())
# print(FacImpl(3))


# print(tf.autograph.to_code(FacImpl))
print(transformed_function.get_concrete_function(tf.constant(1)).graph.as_graph_def())


# tf.summary.trace_on(graph=True, profiler=False)
# # result = transformed_f(tf.constant(3))
# # print(result)
# with writer.as_default():
# 	tf.summary.trace_export("my_function_trace", step=0, profiler_outdir=log_dir)


