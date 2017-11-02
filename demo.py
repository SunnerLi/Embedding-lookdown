import tensorlayer as tl
import tensorflow as tf
import layer_extern
import numpy as np

def load_data():
    seq = np.asarray([[1, 1, 2, 3, 2], [1, 1, 2, 4, 2]])
    return seq

if __name__ == '__main__':
    # Build model
    seq_ph = tf.placeholder(tf.int32, [None, 5])
    network = tl.layers.EmbeddingInputlayer(seq_ph, vocabulary_size=2000, embedding_size = 128, name ='embedding_layer')
    network = layer_extern.EmbeddingReverseLayer(network, embedding_name = 'embedding_layer', name ='reverse_layer')

    # Run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        seq_recover = sess.run(network.outputs, feed_dict={
            seq_ph: load_data()
        })
        print('origin  seq: \n', load_data())
        print('recover seq: \n', seq_recover) 