import tensorflow as tf

#=========================================================================
#
#  This file implements modules for AAE and CycleGANs.      
#
#=========================================================================

w_init = tf.contrib.layers.xavier_initializer()
b_init = tf.constant_initializer(0.)

# w_init = tf.truncated_normal_initializer(stddev=0.01)
# b_init = tf.truncated_normal_initializer(stddev=0.01)


def encoder(inputs, dim_g, encoder_id, keep_prob, nl=tf.nn.leaky_relu, 
            params=None):
    """
    Feed forward encoder
    @param inputs: Input tensor
    @param dim_g: Dimensions of input, hidden and output layers, 
    e.g. [1582, 1000, 1000, 2]
    @param keep_prob: Dropout
    @param nl: Nonlinear activation function
    @param encoder_id: A string to identify the encoder 
    @return layer: Output tensor
    """
    with tf.variable_scope(encoder_id, reuse=tf.AUTO_REUSE):
        layer = inputs
        for i in range(1, len(dim_g)):
            dim_in = dim_g[i-1]
            dim_out = dim_g[i]
            if params is not None:
                w_initializer = tf.constant_initializer(
                    params['encoder/W'+str(i)+':0'])
                b_initializer = tf.constant_initializer(
                    params['encoder/b'+str(i)+':0'])
            else:
                w_initializer = w_init
                b_initializer = b_init
            weight = tf.get_variable(name='W'+str(i),
                                     shape=[dim_in, dim_out],
                                     initializer=w_initializer,
                                     trainable=True)
            bias = tf.get_variable(name='b'+str(i),
                                   shape=[dim_out],
                                   initializer=b_initializer,
                                   trainable=True)
            layer = tf.nn.xw_plus_b(layer, weight, bias)
            if i < len(dim_g)-1:
                layer = nl(layer)
                layer = tf.nn.dropout(layer, keep_prob)
    return layer


def decoder(inputs, dim_g, decoder_id, keep_prob, nl=tf.nn.leaky_relu, 
            params=None):
    """
    Feed forward decoder
    @param inputs: Input tensor
    @param dim_g: Dimensions of output, hidden and input layers, 
    e.g. [1582, 1000, 1000, 2]
    @param keep_prob: Dropout
    @param nl: Nonlinear activation function
    @param decoder_id: A string to identify the decoder 
    @return layer: Output tensor
    """
    with tf.variable_scope(decoder_id, reuse=tf.AUTO_REUSE):
        dim_g_reversed = dim_g[::-1]
        layer = inputs
        for i in range(1, len(dim_g_reversed)):
            dim_in = dim_g_reversed[i-1]
            dim_out = dim_g_reversed[i]
            if params is not None:
                w_initializer = tf.constant_initializer(
                    params['decoder/W'+str(i)+':0'])
                b_initializer = tf.constant_initializer(
                    params['decoder/b'+str(i)+':0'])
            else:
                w_initializer = w_init
                b_initializer = b_init
            weight = tf.get_variable(name='W'+str(i),
                                     shape=[dim_in, dim_out],
                                     initializer=w_initializer,
                                     trainable=True)
            bias = tf.get_variable(name='b'+str(i),
                                   shape=[dim_out],
                                   initializer=b_initializer,
                                   trainable=True)
            layer = tf.nn.xw_plus_b(layer, weight, bias)
            if i < len(dim_g_reversed)-1:
                layer = nl(layer)
                layer = tf.nn.dropout(layer, keep_prob)
    return layer


def discriminator(inputs, dim_d, discriminator_id, keep_prob, 
                  nl=tf.nn.leaky_relu, C=0, params=None):
    """
    Feed forward discriminator
    @param inputs: Input tensor
    @param dim_d: Dimensions of input, hidden and output layers, 
    e.g. [2, 1000, 1000, 1]
    @param keep_prob: Dropout
    @param nl: Nonlinear activation function
    @param C: Number of label classes
    @param discriminator_id: A string to identify the discriminator
    @return layer: Output tensor
    """
    with tf.variable_scope(discriminator_id, reuse=tf.AUTO_REUSE):
        layer = inputs
        for i in range(1, len(dim_d)):
            dim_in = dim_d[i-1]
            if C > 0 and i == 1:
                dim_in += C    # add dimensions for labels               
            dim_out = dim_d[i]
            if params is not None:
                w_initializer = tf.constant_initializer(
                    params['discriminator/W'+str(i)+':0'])
                b_initializer = tf.constant_initializer(
                    params['discriminator/b'+str(i)+':0'])
            else:
                w_initializer = w_init
                b_initializer = b_init
            weight = tf.get_variable(name='W'+str(i),
                                     shape=[dim_in, dim_out],
                                     initializer=w_initializer,
                                     trainable=True)
            bias = tf.get_variable(name='b'+str(i),
                                   shape=[dim_out],
                                   initializer=b_initializer,
                                   trainable=True)
            layer = tf.nn.xw_plus_b(layer, weight, bias)
            if i < len(dim_d)-1:
                layer = nl(layer)
                layer = tf.nn.dropout(layer, keep_prob)
    return layer

