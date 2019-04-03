import tensorflow as tf
import config


def OHNM_single_image(scores, n_pos, neg_mask):
    """Online Hard Negative Mining.
        scores: the scores of being predicted as negative cls
        n_pos: the number of positive samples
        neg_mask: mask of negative samples
        Return:
            the mask of selected negative samples.
            if n_pos == 0, top 10000 negative samples will be selected.
    """

    def has_pos():
        return n_pos * config.max_neg_pos_ratio

    def no_pos():
        return tf.constant(10000, dtype=tf.int32)

    n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
    max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))

    n_neg = tf.minimum(n_neg, max_neg_entries)
    n_neg = tf.cast(n_neg, tf.int32)

    def has_neg():
        neg_conf = tf.boolean_mask(scores, neg_mask)
        vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
        threshold = vals[-1]  # a negtive value
        selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
        return selected_neg_mask

    def no_neg():
        selected_neg_mask = tf.zeros_like(neg_mask)
        return selected_neg_mask

    selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
    return tf.cast(selected_neg_mask, tf.int32)


def OHNM_batch(batch_size, neg_conf, pos_mask, neg_mask):
    selected_neg_mask = []
    for image_idx in xrange(batch_size):
        image_neg_conf = neg_conf[image_idx, :]
        image_neg_mask = neg_mask[image_idx, :]
        image_pos_mask = pos_mask[image_idx, :]
        n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
        selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))

    selected_neg_mask = tf.stack(selected_neg_mask)
    return selected_neg_mask