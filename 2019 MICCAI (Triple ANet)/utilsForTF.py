# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

class Data_set(object):
    def   __init__(self, config, shuffle, name):
        self.tfrecord_file = config.tfdata_path
        self.batch_size = config.batch_size
        self.min_after_dequeue = config.min_after_dequeue
        self.capacity = config.capacity
        self.original_image_size = 320
        self.actual_image_size = config.train_image_size
        self.shuffle = shuffle
        self.name = name

    def read_processing_generate_image_label_batch(self):
        if self.name == 'train':
            # get filename list
            tfrecord_filename = tf.gfile.Glob(self.tfrecord_file + '*%s*' % 'train')
            print('tfrecord train filename', tfrecord_filename)
            filename_queue = tf.train.string_input_producer(tfrecord_filename, num_epochs=None, shuffle=self.shuffle)
            # get tensor of image/label
            image, label = read_tfrecord_and_decode_into_image_label_pair_tensors(filename_queue,
                                                                                  self.original_image_size)
            #image = channels_image_standardization(image)
            image = image_standardization(image)
            #image = tf.image.random_flip_left_right(image)
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=self.batch_size,
                                                              capacity=self.capacity,
                                                              num_threads=2,
                                                              min_after_dequeue=self.min_after_dequeue)

        else:
            # get filename list
            tfrecord_filename = tf.gfile.Glob(self.tfrecord_file + '*%s*' % self.name)
            print('tfrecord test filename', tfrecord_filename)
            # The file name list generator
            filename_queue = tf.train.string_input_producer(tfrecord_filename, num_epochs=None, shuffle=self.shuffle)
            # get tensor of image/label
            image, label = read_tfrecord_and_decode_into_image_label_pair_tensors(filename_queue,
                                                                                  self.original_image_size)
            #image = channels_image_standardization(image)z
            image = image_standardization(image)
            image_batch, label_batch = tf.train.batch([image, label],
                                                              batch_size=self.batch_size,
                                                              capacity=self.capacity)

        image_batch = tf.image.resize_images(image_batch, [self.actual_image_size, self.actual_image_size])

        return image_batch, label_batch

def read_tfrecord_and_decode_into_image_label_pair_tensors(tfrecord_filenames_queue, size):
    """Return label/image tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped label/image tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    Returns
    -------
    image, label : tuple of tf.int32 (image, label)
        Tuple of label/image tensors
    """

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/depth': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            # 'image': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    depth = tf.cast(features['image/depth'], tf.int64)

    image = tf.reshape(image, [size,size,3])   #height,width,depth
    image = tf.to_float(image)

    return image, label

def image_standardization(image):
    out_image = image/255.0
    #out_image = image/127.5 - 1.0
    #out_image = image
    return out_image
