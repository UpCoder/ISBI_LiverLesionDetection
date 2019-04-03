import tensorflow as tf


def loss_with_binary_dice(scores, labels, axis=[1, 2], smooth=1e-5):
    softmaxed = scores[:, :, :, 1]
    print('logits is ', scores)

    cond = tf.less(softmaxed, 0.5)
    output = tf.where(cond, tf.zeros(tf.shape(softmaxed)), tf.ones(tf.shape(softmaxed)))

    target = labels  # tf.one_hot(labels, depth=2)

    if output.get_shape().as_list() != target.get_shape().as_list():
        print output.get_shape().as_list()
        print target.get_shape().as_list()
        assert output.get_shape().as_list() == target.get_shape().as_list()

    with tf.name_scope('dice'):
        output = tf.cast(output, tf.float32)
        target = tf.cast(target, tf.float32)
        inse = tf.reduce_sum(output * target, axis=axis)
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)

    with tf.name_scope('final_cost'):
        # final_cost = ((1 - dice) + (1 / dice) ** 0.1) / 2
        final_cost = (1 - dice) * 1.0

    return final_cost, dice