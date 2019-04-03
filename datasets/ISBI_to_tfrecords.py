# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

# Original dataset organisation.
# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

MEDICAL_LABELS = {
    'none': (0, 'Background'),
    'CYST': (1, 'Begin'),
    'FNH': (1, 'Begin'),
    'HCC': (1, 'Begin'),
    'HEM': (1, 'Begin'),
    'METS': (1, 'Begin'),
}
MEDICAL_LABELS_multi_category = {
    'none': (0, 'Background'),
    'CYST': (1, 'Begin'),
    'FNH': (2, 'Begin'),
    'HCC': (3, 'Begin'),
    'HEM': (4, 'Begin'),
    'METS': (5, 'Begin'),
}
suffix_type = b'JPEG'
suffix_type_name = 'jpg'


def _process_image(img_dir, gt_dir, name, mask_flag=False):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = os.path.join(img_dir, name)
    print('filename is ', filename)
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    if mask_flag:
        mask_filename = os.path.join(os.path.dirname(img_dir), 'mask_gt', name)
        mask_data = tf.gfile.FastGFile(mask_filename, 'rb').read()

    # Read the XML annotation file.
    gt_file_path = os.path.join(gt_dir, name.split('.')[0] + '.txt')
    bboxes = []
    labels = []
    with open(gt_file_path) as gt_file:
        lines = gt_file.readlines()
        for line in lines:
            labels.append(1)
            line_splited = line.split(' ')[1:]
            bboxes.append([float(int(ele) / 512.0) for ele in line_splited])
    if mask_flag:
        return image_data, bboxes, labels, mask_data
    else:
        return image_data, bboxes, labels


def _convert_to_example(image_data, labels, bboxes, mask_data):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    # image_format = b'JPEG'
    image_format = b'PNG'
    if mask_data is None:
        example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': int64_feature(512),
                'image/width': int64_feature(512),
                'image/channels': int64_feature(3),
                'image/shape': int64_feature([512, 512, 3]),
                'image/object/bbox/xmin': float_feature(xmin),
                'image/object/bbox/xmax': float_feature(xmax),
                'image/object/bbox/ymin': float_feature(ymin),
                'image/object/bbox/ymax': float_feature(ymax),
                'image/object/bbox/label': int64_feature(labels),
                'image/format': bytes_feature(image_format),
                'image/encoded': bytes_feature(image_data)}))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(512),
            'image/width': int64_feature(512),
            'image/channels': int64_feature(3),
            'image/shape': int64_feature([512, 512, 3]),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data),
            'maskimage/encoded': bytes_feature(mask_data)}))
    return example


def _add_to_tfrecord(img_dir, gt_dir, name, tfrecord_writer, mask_flag=False):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    if mask_flag:
        image_data, bboxes, labels, mask_data = _process_image(img_dir, gt_dir, name, mask_flag)
    else:
        image_data, bboxes, labels = _process_image(img_dir, gt_dir, name, mask_flag)
        mask_data = None
    example = _convert_to_example(image_data, labels, bboxes, mask_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, shuffling=True, name='medicalimage', mask_flag=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, 'img')
    gt_dir = os.path.join(dataset_dir, 'gt')
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()
                filename = filenames[i]
                _add_to_tfrecord(os.path.join(dataset_dir, 'img'), gt_dir, filename, tfrecord_writer, mask_flag)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')


if __name__ == '__main__':
    dataset_dir = '/home/give/Documents/dataset/ISBI2017/Detection_V4/Batch_2'
    output_dir = '/home/give/Documents/dataset/ISBI2017/Detection_V4/Batch_2/tfrecords'

    run(dataset_dir, output_dir, mask_flag=True)
