from tensorlayer.layers import Layer
import tensorlayer as tl
import tensorflow as tf

class EmbeddingInputlayer(Layer):
    """
    The :class:`EmbeddingInputlayer` class is a fully connected layer,
    for Word Embedding. Words are input as integer index.
    The output is the embedded word vector.

    If you have a pre-train matrix, you can assign the matrix into it.
    To train a word embedding matrix, you can used class:`Word2vecEmbeddingInputlayer`.

    Note that, do not update this embedding matrix.

    Parameters
    ----------
    inputs : placeholder
        For word inputs. integer index format.
        a 2D tensor : [batch_size, num_steps(num_words)]
    vocabulary_size : int
        The size of vocabulary, number of words.
    embedding_size : int
        The number of embedding dimensions.
    E_init : embedding initializer
        The initializer for initializing the embedding matrix.
    E_init_args : a dictionary
        The arguments for embedding initializer
    name : a string or None
        An optional name to attach to this layer.

    Variables
    ------------
    outputs : a tensor
        The outputs of embedding layer.
        the outputs 3D tensor : [batch_size, num_steps(num_words), embedding_size]

    Examples
    --------
    >>> vocabulary_size = 50000
    >>> embedding_size = 200
    >>> model_file_name = "model_word2vec_50k_200"
    >>> batch_size = None
    ...
    >>> all_var = tl.files.load_npy_to_any(name=model_file_name+'.npy')
    >>> data = all_var['data']; count = all_var['count']
    >>> dictionary = all_var['dictionary']
    >>> reverse_dictionary = all_var['reverse_dictionary']
    >>> tl.files.save_vocab(count, name='vocab_'+model_file_name+'.txt')
    >>> del all_var, data, count
    ...
    >>> load_params = tl.files.load_npz(name=model_file_name+'.npz')
    >>> x = tf.placeholder(tf.int32, shape=[batch_size])
    >>> y_ = tf.placeholder(tf.int32, shape=[batch_size, 1])
    >>> emb_net = tl.layers.EmbeddingInputlayer(
    ...                inputs = x,
    ...                vocabulary_size = vocabulary_size,
    ...                embedding_size = embedding_size,
    ...                name ='embedding_layer')
    >>> tl.layers.initialize_global_variables(sess)
    >>> tl.files.assign_params(sess, [load_params[0]], emb_net)
    >>> word = b'hello'
    >>> word_id = dictionary[word]
    >>> print('word_id:', word_id)
    ... 6428
    ...
    >>> words = [b'i', b'am', b'hao', b'dong']
    >>> word_ids = tl.files.words_to_word_ids(words, dictionary)
    >>> context = tl.files.word_ids_to_words(word_ids, reverse_dictionary)
    >>> print('word_ids:', word_ids)
    ... [72, 1226, 46744, 20048]
    >>> print('context:', context)
    ... [b'i', b'am', b'hao', b'dong']
    ...
    >>> vector = sess.run(emb_net.outputs, feed_dict={x : [word_id]})
    >>> print('vector:', vector.shape)
    ... (1, 200)
    >>> vectors = sess.run(emb_net.outputs, feed_dict={x : word_ids})
    >>> print('vectors:', vectors.shape)
    ... (4, 200)

    """
    def __init__(
        self,
        inputs = None,
        vocabulary_size = 80000,
        embedding_size = 200,
        E_init = tf.random_uniform_initializer(-0.1, 0.1),
        E_init_args = {},
        name ='embedding_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = inputs
        print("  [TL] EmbeddingInputlayer %s: (%d, %d)" % (self.name, vocabulary_size, embedding_size))

        with tf.variable_scope(name, reuse=False) as vs:
            tl.layers.set_name_reuse(False)
            embeddings = tf.get_variable(name='embeddings',
                                    shape=(vocabulary_size, embedding_size),
                                    initializer=E_init,
                                    **E_init_args)
            embed = tf.nn.embedding_lookup(embeddings, self.inputs)

        self.outputs = embed

        self.all_layers = [self.outputs]
        self.all_params = [embeddings]
        self.all_drop = {}

class EmbeddingReverseLayer(tl.layers.Layer):
    def __init__(
        self,
        layer = None,
        vocabulary_size = 80000,
        embedding_size = 200,
        embedding_name = None,
        E_init = tf.random_uniform_initializer(-0.1, 0.1),
        E_init_args = {},
        name ='embedding_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        # Get origin embedded
        with tf.variable_scope(embedding_name, reuse=True) as vs:
            tl.layers.set_name_reuse(True)
            origin_embeddings = tf.get_variable(name='embeddings', **E_init_args)
            vocabulary_size = int(origin_embeddings.get_shape()[0])
            embedding_size = int(origin_embeddings.get_shape()[1])
        print("  [TL] EmbeddingReverselayer %s: (%d, %d)" % (self.name, vocabulary_size, embedding_size))

        """
            self.inputs = [None, 5, 128]
            embed       = [2000, 128]
            [None, 5, 2000, 128] - [None, 5, 1, 128]
        """

        # Create range to lookup
        with tf.variable_scope(name) as vs:
            dummy_index = tf.range(0, limit=vocabulary_size)
            embed = tf.nn.embedding_lookup(origin_embeddings, dummy_index)

            print('1 size: ', self.inputs.get_shape()[1])

            embed_reshape = tf.reshape(embed, [-1])
            embed_tile = tf.tile(embed_reshape, [tf.shape(self.inputs)[0] * tf.shape(self.inputs)[1]])
            embed_tile_reshape = tf.reshape(embed_tile, [tf.shape(self.inputs)[0], int(self.inputs.get_shape()[1]), vocabulary_size, embedding_size])
            input_embed = tf.expand_dims(self.inputs, axis=2)

            print(embed_tile.get_shape())
            print(input_embed.get_shape())

            embed_diff = embed_tile_reshape - input_embed
            embed_distance = tf.reduce_sum(tf.square(embed_diff), axis=-1)
            embed_prob = tf.nn.softmax(-embed_distance + tf.reduce_min(embed_distance), dim=-1)
            embed_index = tf.arg_max(embed_prob, -1)

            """
            embed_tile = tf.reshape(tf.tile(tf.reshape(embed, [-1]), [int(self.inputs.get_shape()[1])]), [-1, vocabulary_size, embedding_size])
            embed_distance = tf.reduce_sum(tf.square(embed_tile - self.inputs), axis=-1)
            embed_prob = tf.nn.softmax(-embed_distance + tf.reduce_min(embed_distance), dim=-1)
            embed_index = tf.arg_max(embed_prob, -1)
            """

        self.outputs = embed_index

        self.all_layers = [self.outputs]
        self.all_params = [dummy_index, embed, embed_distance, embed_prob, embed_index]
        self.all_drop = {}