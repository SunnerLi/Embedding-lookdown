import tensorflow as tf

with tf.Graph().as_default():  
    a = tf.constant([[1,2,3],[4,5,6]],name='a')     
    b = tf.reshape(a, [-1])
    c = tf.tile(b, [6])
    d = tf.reshape(c, [2, 3, 2, 3])
    sess = tf.Session()
    print(sess.run(d))