import scipy.misc
import numpy as np
import os
from glob import glob
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from keras.datasets import cifar10, mnist
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from tensorflow.contrib.layers import batch_norm
from tflearn.layers.conv import global_avg_pool


def get_image_label_batch(batch_size, shuffle, name):
    with tf.name_scope('get_batch'):
        Data = Data_set(batch_size = batch_size, shuffle=shuffle, name=name)
        image_batch, label_batch = Data.read_processing_generate_image_label_batch(batch_size)
    return image_batch, label_batch


class Data_set(object):
    def   __init__(self, batch_size, shuffle, name):
        self.tfrecord_file = '/home/xxh/Documents/Journal/dataset/sum_train_test_only/split1/'
        self.min_after_dequeue = 100
        self.capacity = 200
        self.original_size = 320
        self.actual_image_size = 227
        self.shuffle = shuffle
        self.name = name

    def read_processing_generate_image_label_batch(self, batch_size):
        if self.name == 'train':
            # get filename list
            tfrecord_filename = tf.gfile.Glob(self.tfrecord_file + '*%s*' % 'train')
            print('tfrecord train filename', tfrecord_filename)
            filename_queue = tf.train.string_input_producer(tfrecord_filename, num_epochs=None, shuffle=self.shuffle)
            # get tensor of image/label
            image, label = read_tfrecord_and_decode_into_image_label_pair_tensors(filename_queue,
                                                                                  self.original_size)
            #image = channels_image_standardization(image)
            image = image_standardization(image)
            #image = tf.image.random_flip_left_right(image)
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=batch_size,
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
                                                                                  self.original_size)
            #image = channels_image_standardization(image)z
            image = image_standardization(image)
            image_batch, label_batch = tf.train.batch([image, label],
                                                              batch_size=batch_size,
                                                              capacity=self.capacity)
                                                              # num_threads=self.num_threads)
        image_batch = tf.image.resize_images(image_batch, [self.actual_image_size, self.actual_image_size])
        return image_batch, label_batch

def read_tfrecord_and_decode_into_image_label_pair_tensors(tfrecord_filenames_queue, size):
    
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

    image = tf.to_float(image)
    image = tf.reshape(image, [size,size,3])   #height,width,depth
    label = tf.one_hot(label, 3)
    return image, label

def image_standardization(image):
    out_image = image/255.0
    return out_image
