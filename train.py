import tensorflow as tf
import network
import evaluate

classes = 23

X = tf.compat.v1.placeholder(dtype = tf.float32,shape = [None, None, None, 3])
Y = tf.compat.v1.placeholder(dtype = tf.float32,shape = [None, None, None, classes])
last_stage2 = tf.compat.v1.placeholder(dtype = tf.float32,shape = [None, None, None, 96])
last_stage3 = tf.compat.v1.placeholder(dtype = tf.float32,shape = [None, None, None, classes])
training = tf.compat.v1.placeholder(dtype=tf.bool)
update_3 = tf.compat.v1.placeholder(dtype=tf.bool)
update_2 = tf.compat.v1.placeholder(dtype=tf.bool)

prediction, this_stage2, this_stage3 = network.clock_shufflenet(X,training,classes,update_3,update_2,last_stage2,last_stage3)

cost = evaluate.xentropy(prediction,Y)

update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cost)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    #TODO: The rest of the training code, including
    #   1. Training
    #   2. Output training result
    #   3. Save the trained model