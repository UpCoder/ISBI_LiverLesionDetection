# -*- coding=utf-8 -*-
from visualize_detection_result import draw_bbox


def learn_random_brightness():
    import tensorflow as tf
    import numpy as np
    img = tf.ones([512, 512, 3], dtype=tf.float32)
    random_tensor = tf.random_uniform([])
    multiply_tensor = random_tensor * random_tensor
    add_tensor = random_tensor + random_tensor
    seed = np.random.randint(1, 100)
    brightness_image_tensor1 = tf.image.random_brightness(img, max_delta=32. / 255., seed=seed)
    brightness_image_tensor2 = tf.image.random_brightness(img, max_delta=32. / 255., seed=seed)
    with tf.Session() as sess:
        # brightness_image1 = sess.run(brightness_image_tensor1)
        # print(np.sum(brightness_image1))
        print(sess.run([random_tensor, multiply_tensor, add_tensor]))
        for i in range(3):
            brightness_image1, brightness_image2 = sess.run([brightness_image_tensor1, brightness_image_tensor2])
            print(np.sum(brightness_image1), np.sum(brightness_image2))


def learn_control_dependency():
    import tensorflow as tf
    a = tf.Variable(0.0)
    # add_op = tf.assign_add(a, 1.0)  # 该operation每执行一次，a就增加1 返回的是a的一个引用
    # add_tensor_1 = tf.identity(a) + 1.0
    with tf.control_dependencies([tf.assign_add(a, 1.0)]):
        add_tensor = a + 1.0


    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))
        for idx in range(3):
            print(sess.run([add_tensor]))


def learn_dataset():
    import tensorflow as tf
    import numpy as np
    from train_pixel_link import config_initialization, create_dataset_batch_queue
    from tensorflow.contrib import slim
    import config
    import pixel_link
    dataset = config_initialization()
    batch_queue = create_dataset_batch_queue(dataset, preprocessing_flag=False)
    b_image, b_pixel_cls_label, b_pixel_cls_weight,\
    b_pixel_link_label, b_pixel_link_weight = batch_queue.dequeue()
    import cv2
    import os
    tmp_dir = os.path.join('./', 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        print('ok')
        # 使用start_queue_runners 启动队列填充
        threads = tf.train.start_queue_runners(sess, coord)
        count = 0
        for idx in range(100):
            b_image_v, b_pixel_cls_label_v, b_pixel_cls_weight_v,\
            b_pixel_link_label_v, b_pixel_link_weight_v = sess.run(
                [b_image, b_pixel_cls_label, b_pixel_cls_weight, b_pixel_link_label, b_pixel_link_weight])
            for idx, (image, mask_label) in enumerate(zip(b_image_v, b_pixel_cls_label_v)):
                cv2.imwrite('/home/give/PycharmProjects/ISBI_Detection/tmp/%d.png' % count, np.asarray(image, np.uint8)[:, :, 1])
                print np.shape(mask_label * 100)
                cv2.imwrite('/home/give/PycharmProjects/ISBI_Detection/tmp/%d_mask.png' % count,
                            cv2.resize(mask_label * 100, (512, 512), interpolation=cv2.INTER_NEAREST))
                count += 1
            print(np.shape(b_image_v))


def learn_read_png():
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    filename_queue = tf.train.string_input_producer([
     '/home/give/Documents/dataset/LiverLesionDetection_Splited/JPG/0/NCARTPV_tripleslice_mask/train_mask/034-3080765-1--2-3.PNG'])  # list of files to read

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    my_img = tf.image.decode_image(value, channels=1)  # use png or jpg decoder based on your files.

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        # Start populating the filename queue.

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1):  # length of your filename list
            image = my_img.eval()  # here is your image Tensor :)
        print(np.unique(image))
        print(image.shape, np.max(image), np.min(image))
        img = Image.fromarray(np.asarray(np.squeeze(image) * 50, np.uint8))
        img.show()
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    learn_dataset()
    # learn_read_png()