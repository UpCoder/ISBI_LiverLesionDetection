import tensorflow as tf

slim = tf.contrib.slim


def basenet(inputs, fatness = 64, dilation = True, pooling='MAX'):
    """
    backbone net of vgg16
    """
    if pooling == 'MAX':
        pooling_func = slim.max_pool2d
    else:
        pooling_func = slim.avg_pool2d
    # End_points collect relevant activations for external use.
    end_points = {}
    # Original VGG-16 blocks.
    with slim.arg_scope([slim.conv2d, pooling_func], padding='SAME'):
        # Block1
        net = slim.repeat(inputs, 2, slim.conv2d, fatness, [3, 3], scope='conv1')
        end_points['conv1_2'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = pooling_func(net, [2, 2], scope='pool1')
        end_points['pool1'] = net
        
        
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, fatness * 2, [3, 3], scope='conv2')
        end_points['conv2_2'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = pooling_func(net, [2, 2], scope='pool2')
        end_points['pool2'] = net
        
        
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 4, [3, 3], scope='conv3')
        end_points['conv3_3'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = pooling_func(net, [2, 2], scope='pool3')
        end_points['pool3'] = net
        
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv4')
        end_points['conv4_3'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = pooling_func(net, [2, 2], scope='pool4')
        end_points['pool4'] = net
        
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv5')
        end_points['conv5_3'] = net
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')
        net = pooling_func(net, [3, 3], 1, scope='pool5')
        end_points['pool5'] = net

        # fc6 as conv, dilation is added
        if dilation:
            net = slim.conv2d(net, fatness * 16, [3, 3], rate=6, scope='fc6')
        else:
            net = slim.conv2d(net, fatness * 16, [3, 3], scope='fc6')
        end_points['fc6'] = net

        # fc7 as conv
        net = slim.conv2d(net, fatness * 16, [1, 1], scope='fc7')
        end_points['fc7'] = net

    return net, end_points


if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, [None, 512, 512, 3])
    net, end_points = basenet(input_tensor)
    keys = end_points.keys()
    keys.sort()
    for key in keys:
        print key, end_points[key]


