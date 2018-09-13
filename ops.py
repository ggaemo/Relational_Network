import tensorflow as tf

def conv(x, channels, kernel, stride, norm,
         activation, is_training):

    # if pad_type == 'zero' :
    # x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    # if pad_type == 'reflect' :
    #     x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

    x = tf.layers.conv2d(inputs=x, filters=channels,
                         kernel_size=kernel,
                         strides=stride, padding='same')
    if norm == 'bn':
        x = tf.layers.batch_normalization(x, training=is_training)
    elif norm == 'in':
        x = tf.contrib.layers.instance_norm(x,epsilon=1e-05,
                                            center=True, scale=True)
    elif norm == 'ln':
        x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
    elif norm == 'none':
        x = x
    else:
        gamma, beta = norm
        epsilon = 1e-5

        c_mean, c_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        c_std = tf.sqrt(c_var + epsilon)

        x = gamma * ((x - c_mean) / c_std) + beta

    if activation:
        x = activation(x)

    return x