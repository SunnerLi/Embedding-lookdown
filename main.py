import tensorlayer as tl
import tensorflow as tf
import numpy as np
import layer

def load_data():
    seq = np.asarray([1, 1, 2, 3, 2])
    return seq

if __name__ == '__main__':
    # Build model
    seq_ph = tf.placeholder(tf.int32, [None])
    network = tl.layers.EmbeddingInputlayer(seq_ph, vocabulary_size=2000, embedding_size = 128, name ='embedding_layer')
    # network = tl.layers.DenseLayer(network, n_units = 128)
    network = layer.EmbeddingReverseLayer(network, embedding_name = 'embedding_layer', name ='reverse_layer')

    # Run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _ = sess.run(network.outputs, feed_dict={
            seq_ph: load_data()
        })