#encoding = utf-8

import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.training.python.training import evaluation
from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import metrics as tfe_metrics
import util
import cv2
import pixel_link
from nets import pixel_link_symbol


slim = tf.contrib.slim
import config
# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints\
    in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
  'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')


# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_bool('preprocessing_use_rotation', False, 
             'Whether to use rotation for data augmentation')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'icdar2015', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string('dataset_dir', 
           util.io.get_absolute_path('~/dataset/ICDAR2015/Challenge4/ch4_test_images'), 
           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer('eval_image_width', 1280, 'Train image size')
tf.app.flags.DEFINE_integer('eval_image_height', 768, 'Train image size')
tf.app.flags.DEFINE_bool('using_moving_average', True, 
                         'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 
                          'The decay rate of ExponentionalMovingAverage')
tf.app.flags.DEFINE_string('pred_path', '', '')
tf.app.flags.DEFINE_bool('modify_flag', False, '')
tf.app.flags.DEFINE_string('score_map_path', '', '')

FLAGS = tf.app.flags.FLAGS


def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # config.load_config(FLAGS.checkpoint_path)
    # config.load_config(util.io.get_dir(FLAGS.checkpoint_path))
    config.load_config('/home/give/PycharmProjects/ISBI_Detection')
    config.init_config(image_shape, 
                       batch_size = 1, 
                       pixel_conf_threshold = 0.5,
                       link_conf_threshold = 0.1,
                       num_gpus = 1, 
                   )
    util.proc.set_proc_name('test_pixel_link_on'+ '_' + FLAGS.dataset_name)


def to_txt(txt_path, image_name, 
           image_data, pixel_pos_scores, link_pos_scores):
    # write detection result as txt files
    def write_result_as_txt(image_name, bboxes, path):
        filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
        lines = []
        for b_idx, bbox in enumerate(bboxes):
              values = [int(v) for v in bbox]
              line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
              lines.append(line)
        util.io.write_lines(filename, lines)
        print 'result has been written to:', filename

    def write_result_as_pred_txt(image_name, bboxes, bboxes_score):
        filename = util.io.join_path(FLAGS.pred_path, '%s.txt' % image_name)
        lines = []
        for b_idx, (bbox, bbox_score) in enumerate(zip(bboxes, bboxes_score)):
            min_x = np.min([bbox[0], bbox[2], bbox[4], bbox[6]])
            max_x = np.max([bbox[0], bbox[2], bbox[4], bbox[6]])
            min_y = np.min([bbox[1], bbox[3], bbox[5], bbox[7]])
            max_y = np.max([bbox[1], bbox[3], bbox[5], bbox[7]])
            lines.append('%s %.4f %d %d %d %d\n' % ('Tumor', bbox_score, min_x, min_y, max_x, max_y))
        util.io.write_lines(filename, lines)
        print('result has been written to: ', filename)
    # mask = pixel_link.decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
    # bboxes, bboxes_score = pixel_link.mask_to_bboxes(mask, pixel_pos_scores, image_data.shape)

    # new version---simples
    pixel_pos_label = np.asarray(pixel_pos_scores > config.pixel_conf_threshold, np.uint8)
    pixel_pos_label = np.squeeze(pixel_pos_label)
    print(np.shape(pixel_pos_label))
    from skimage.measure import label
    label_res = label(pixel_pos_label, neighbors=8)
    bboxes = []
    bboxes_score = []
    for i in range(1, np.max(label_res)+1):
        cur_label_map = np.asarray(label_res == i, np.uint8)
        xs, ys = np.where(cur_label_map == 1)
        min_ys = np.min(xs)
        max_ys = np.max(xs)
        min_xs = np.min(ys)
        max_xs = np.max(ys)
        bboxes.append([min_xs, min_ys, min_xs, max_ys, max_xs, max_ys, max_xs, min_ys])

        bboxes_score.append(np.mean(pixel_pos_scores[0, xs, ys]))

    write_result_as_txt(image_name, bboxes, txt_path)
    write_result_as_pred_txt(image_name.split('.')[0], bboxes, bboxes_score=bboxes_score)
    if not util.io.exists(FLAGS.score_map_path):
        util.io.mkdir(FLAGS.score_map_path)
    score_map_path = util.io.join_path(FLAGS.score_map_path, image_name+'.png')
    print('score_map_path: ', score_map_path)
    print(
    np.shape(np.transpose(np.asarray(pixel_pos_scores * 255, np.uint8), axes=[2, 1, 0])), np.max(pixel_pos_scores))
    cv2.imwrite(score_map_path, np.squeeze(np.transpose(np.asarray(pixel_pos_scores * 255, np.uint8), axes=[2, 1, 0])))


def test():
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
        image_shape = tf.placeholder(dtype = tf.int32, shape = [3, ])
        processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None, 
                                                   out_shape = config.image_shape,
                                                   data_format = config.data_format, 
                                                   is_training = False)
        b_image = tf.expand_dims(processed_image, axis = 0)
        modify_flag = FLAGS.modify_flag
        if modify_flag:
            print("modify_version")
            net = pixel_link_symbol.PixelLinkNetModify(b_image, is_training=True, mask_input=None)
        else:
            print("original version")
            net = pixel_link_symbol.PixelLinkNet(b_image, is_training = True)
        global_step = slim.get_or_create_global_step()

    
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    checkpoint_dir = util.io.get_dir(FLAGS.checkpoint_path)
    logdir = util.io.join_path(checkpoint_dir, 'test', FLAGS.dataset_name + '_' +FLAGS.dataset_split_name)

    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()
    
    saver = tf.train.Saver(var_list = variables_to_restore)
    
    
    image_names = util.io.ls(FLAGS.dataset_dir)
    image_names.sort()
    
    checkpoint = FLAGS.checkpoint_path
    checkpoint_name = util.io.get_filename(str(checkpoint));
    dump_path = util.io.join_path(logdir, checkpoint_name)
    txt_path = util.io.join_path(dump_path,'txt')        
    zip_path = util.io.join_path(dump_path, checkpoint_name + '_det.zip')
    
    with tf.Session(config = sess_config) as sess:
        saver.restore(sess, checkpoint)

        for iter, image_name in enumerate(image_names):
            image_data = util.img.imread(
                util.io.join_path(FLAGS.dataset_dir, image_name), rgb = True)
            image_name = image_name.split('.')[0]
            pixel_pos_scores, link_pos_scores = sess.run(
                [net.pixel_pos_scores, net.link_pos_scores], 
                feed_dict = {
                    image:image_data
            })
               
            print '%d/%d: %s'%(iter + 1, len(image_names), image_name)
            to_txt(txt_path,
                    image_name, image_data, 
                    pixel_pos_scores, link_pos_scores)


            
    # create zip file for icdar2015
    cmd = 'cd %s;zip -j %s %s/*'%(dump_path, zip_path, txt_path);
    print cmd
    util.cmd.cmd(cmd);
    print "zip file created: ", util.io.join_path(dump_path, zip_path)

         

def main(_):
    config_initialization()
    test()
                
    
if __name__ == '__main__':
    tf.app.run()
