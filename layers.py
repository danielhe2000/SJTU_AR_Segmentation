import tensorflow as tf


def conv(inputs, name, stride, in_channel, out_channel, k_size, training):
    """A normal convolution layer
    Args:
        As name suggests. Training indicates whether we are in the process of training or inference
    Returns:
        A batch normalized, ReLU'ed convolution of input
    """
    with tf.name_scope(name) as scope:
        kernel = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([k_size,k_size,in_channel,out_channel]),
                                       trainable = True, name = 'k')
        convolution = tf.nn.conv2d(inputs,kernel,strides=[1,stride,stride,1],padding='SAME')
        bn_result = tf.compat.v1.layers.batch_normalization(convolution,training=training)
        output_layer = tf.nn.relu(bn_result)

        return output_layer


def dpw_conv(inputs, name, stride, in_channel, training):
    """A depth wise convolution layer
        Args:
            As name suggests. Training indicates whether we are in the process of training or inference
        Returns:
            A batch normalized convolution of input
    """
    with tf.name_scope(name) as scope:
        kernel = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([3,3,in_channel,1]), trainable=True, name = 'dpw_k')

        convolution = tf.nn.depthwise_conv2d(inputs,kernel,strides=stride,padding='SAME')
        return tf.compat.v1.layers.batch_normalization(convolution,training=training)


def pool(inputs, name, stride, k_size):
    """A normal pooling layer
        Args:
            As name suggests
        Returns:
            A max pooled  input
    """
    return tf.nn.max_pool(inputs,ksize=[1,k_size,k_size,1],stride = [1,stride,stride,1], padding = 'SAME', name=name)


def channle_shuffle(inputs, group):
    """Shuffle the channel
    Args:
        inputs: 4D Tensor
        group: int, number of groups
    Returns:
        Shuffled 4D Tensor
    """
    in_shape = inputs.get_shape().as_list()
    h, w, in_channel = in_shape[1:]
    assert in_channel % group == 0
    l = tf.reshape(inputs, [-1, h, w, in_channel // group, group])
    l = tf.transpose(l, [0, 1, 2, 4, 3])
    l = tf.reshape(l, [-1, h, w, in_channel])

    return l


def shuffle_net_unit(inputs, name, in_channel, training):
    """A shuffle net unit
        Args:
            As name suggests. Training indicates whether we are in the process of training or inference
        Returns:
            A shuffle net unit result with the same dimension as input
    """
    shortcut, x = tf.split(inputs,2,axis=3)
    x = conv(x,name=name+'c1',stride=1,in_channel=in_channel/2,out_channel=in_channel,k_size=1,training=training)
    x = dpw_conv(x,name = name + 'dpw', stride=1, in_channel=in_channel/2, training=training)
    x = conv(x,name=name+'c2',stride=1,in_channel=in_channel/2,out_channel=in_channel,k_size=1,training=training)

    x = tf.concat([shortcut, x], axis=3)
    x = channle_shuffle(x,2)

    return x


def down_sample_unit(inputs, name, in_channel, training):
    """A down sample unit of the shuffle net
        Args:
            As name suggests. Training indicates whether we are in the process of training or inference
        Returns:
            A down sample result with one half the size and twice the channel as input
    """
    x = conv(inputs,name=name+'cr1',stride=1,in_channel=in_channel,out_channel=in_channel,k_size=1,training=training)
    x = dpw_conv(x,name = name + 'dpw_r', stride=2, in_channel=in_channel, training=training)
    x = conv(x,name=name+'cr2',stride=1,in_channel=in_channel,out_channel=in_channel,k_size=1,training=training)

    y = dpw_conv(inputs,name = name + 'dpw_l', stride=2, in_channel=in_channel, training=training)
    y = conv(y,name=name+'cr2',stride=1,in_channel=in_channel,out_channel=in_channel,k_size=1,training=training)

    output_layer = tf.concat([y,x], axis=3)
    output_layer = channle_shuffle(output_layer,2)

    return output_layer


def up_sample_unit(inputs, name, stride, num_classes):
    """A down sample unit of the shuffle net
        Args:
            As name suggests. 
        Returns:
            An up sample result with the same channel as input and "stride" size of input
    """
    with tf.name_scope(name) as scope:

        in_shape = tf.shape(inputs)

        k_size = stride*2

        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = [in_shape[0], h, w, num_classes]
        output_shape = tf.stack(new_shape)

        kernel = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([k_size,k_size,num_classes,num_classes]),
                                       trainable = True, name = 'k')
        convolution = tf.nn.conv2d_transpose(inputs,kernel,output_shape=output_shape,strides=[1,stride,stride,1],padding='SAME')

        return convolution
