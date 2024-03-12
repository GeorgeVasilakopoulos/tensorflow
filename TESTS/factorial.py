import tensorflow as tf


# Verbosity is now 5

log_dir = "./graph"

tf.config.run_functions_eagerly(False)
writer = tf.summary.create_file_writer(log_dir)
# tf.autograph.set_verbosity(5,True)

# @tf.function
# def fact(x):
# 	if(x == 1):
# 		return 1
# 	else:
# 		return x * fact(x-1)


def fact_loop(n):
	i = tf.constant(1)
	result = tf.constant(1)

	# Loop condition: continue while i <= n
	def cond(i, result):
		return i <= n

	# Loop body: multiply result by i and increment i
	def body(i, result):
		return i + 1, result * i

	# Execute the while loop
	_, final_result = tf.while_loop(cond, body, loop_vars=[i, result])
	return final_result

# def fact(n):
# 	condition = tf.equal(n, tf.constant(1))
# 	true_branch = tf.constant(1)
# 	subtraction_op = tf.subtract(n, 1)
# 	false_branch = tf.multiply(fact(subtraction_op), n)
# 	return tf.cond(condition, lambda: true_branch, lambda: false_branch)

transformed_f = tf.function(fact_loop)

result = transformed_f(3)
print(result)
# print(tf.autograph.to_code(transformed_f))
# graph_f = tf.autograph.to_graph(fact)


# print(graph_f)




tf.summary.trace_on(graph=True, profiler=False)
# result = transformed_f(tf.constant(3))
# print(result)
with writer.as_default():
	tf.summary.trace_export("my_function_trace", step=0, profiler_outdir=log_dir)
