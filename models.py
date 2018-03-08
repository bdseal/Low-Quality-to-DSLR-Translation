import tensorflow as tf
import tensorflow.contrib.slim as slim

def resnet(input_image):
    '''
    https://github.com/jmiller656/EDSR-Tensorflow/blob/master/model.py
    EDSR paper: num_layers=32,feature_size=256 
    dped paper: num_layers=4,feature_size=64
    '''
    with tf.variable_scope("generator"):

        #Preprocessing as mentioned in the EDSR paper, by subtracting the mean  of the entire dataset. Here subtracting the mean of each batch.
        mean_input_image=tf.reduce_mean(input_image)
        image_input=input_image-mean_input_image

        num_layers=32
        feature_size=64
        output_channels=3
        w= weight_variable([9, 9, 36, feature_size]); b = bias_variable([feature_size]);
        x = conv2d(input_image,w )+b
        # Store the output of the first convolution to add later
        conv_1 = x

        for i in range(num_layers):
            # x = ResBlock_NoBN(x)
            x = ResBlock(x)

        #three more convolution, and then we add the output of our first conv layer
        w= weight_variable([3, 3, feature_size, feature_size]); b = bias_variable([feature_size]);
        x = conv2d(x, w)+b

        w= weight_variable([3, 3, feature_size, feature_size]); b = bias_variable([feature_size]);
        x = conv2d(x, w)+b

        x += conv_1
        w= weight_variable([9, 9, feature_size, 3]); b = bias_variable([3]);
        x = conv2d(x, w)+b      

        # Final
 
        # enhanced = tf.clip_by_value(x,0.0,1.0)
        enhanced = tf.nn.tanh(x) * 0.5 + 0.5

    return enhanced

def adversarial(image_):

    with tf.variable_scope("discriminator"):
        #_conv_layer(net, num_filters, filter_size, strides, batch_nn=True)

        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)
        
        flat_size = 128 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out

"""
Creates a convolutional residual block as defined in the paper. More on this inside model.py
x: input to pass through the residual block
channels: number of channels to compute
stride: convolution stride
"""

def ResBlock(x):
    w= weight_variable([3, 3, 64, 64]); b = bias_variable([64])
    tmp = tf.nn.relu(_instance_norm(conv2d(x, w)+b)) 

    w= weight_variable([3, 3, 64, 64]); b = bias_variable([64])
    tmp = tf.nn.relu(_instance_norm(conv2d(tmp, w)+b)) 
    return x + tmp

def ResBlock_NoBN(x,channels=64,kernel_size=[3,3]):
    scale=0.1
    tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
    tmp*=scale
    return x + tmp

def weight_variable(shape, name='w'):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name='b'):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias   
    net = leaky_relu(net)

    if batch_nn:
        net = _instance_norm(net)

    return net

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init
