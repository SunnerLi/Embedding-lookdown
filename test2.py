import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[1, 1]])
c = a-b
sess = tf.Session()
print(sess.run(c))