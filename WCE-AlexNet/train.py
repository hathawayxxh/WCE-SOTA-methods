"""
This is the network used by Fan et al.
The paper "Computer-aided detection of small intestinal ulcer and erosion in wireless capsule endoscopy images" 
is published at Physics in Medicine and Biology.
They used the Alexnet directly.
"""

import sys
import os
import cv2
import time
import shutil
import math
from ops import *
from skimage import io
from datetime import timedelta
from sklearn import metrics
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
import  pandas as pd # used to write and read csv files.
from alexnet import AlexNet

import data_provider
from data_provider import get_image_label_batch
"""
Configuration settings
"""
# Learning params
learning_rate = 0.001
n_epochs = 100
batch_size = 32

num_train = 9688
num_test = 2400
# num_train = 32

# Network params
dropout_rate = 0.5
num_classes = 3

batches_step = 0

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool, shape=[])

# Initialize model
model = AlexNet(x, keep_prob, num_classes)

# Link variable to model output
score = model.fc8
prediction = tf.nn.softmax(score)

att_maps = model.att_map

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables()]

loss_L2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list if 'biases' not in v.name]) * 0.001

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  net_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y) + loss_L2)  
  
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True)
train_op = optimizer.minimize(net_loss, var_list = var_list)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Start Tensorflow session
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
logswriter = tf.summary.FileWriter
summary_writer = logswriter('./logs')


def log_loss_accuracy(loss, accuracy, epoch, prefix, should_print=True):
    if should_print:
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))

    summary = tf.Summary(value=[
        tf.Summary.Value(tag='loss_%s' % prefix, simple_value=float(loss)),
        tf.Summary.Value(tag='accuracy_%s' % prefix, simple_value=float(accuracy))
    ])
    summary_writer.add_summary(summary, epoch)


def save_model(global_step=None):
    saver = tf.train.Saver()
    save_path = 'save_model/split1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'model.chkpt')
    saver.save(sess, save_path, global_step=global_step)

def load_model():
    saver = tf.train.Saver()
    save_path = 'save_model/split1/model.chkpt'
    try:
        saver.restore(sess, save_path)
    except Exception as e:
        raise IOError("Failed to load model from save path: %s" % save_path)
    saver.restore(sess, save_path)
    print("Successfully load model from save path: %s" % save_path)


def train_all_epochs(learning_rate):

    total_start_time = time.time()

    loss_all_epochs = []
    acc_all_epochs = []

    nr_all_epochs = []
    br_all_epochs = []
    ir_all_epochs = []
    kappa_all_epochs = []

    best_acc = 0.0

    train_image_batch, train_label_batch = get_image_label_batch(batch_size, shuffle=True, name='train')
    test_image_batch, test_label_batch = get_image_label_batch(batch_size, shuffle=False, name='4aug_test')

    print('Train image shape:', train_image_batch.shape)
    print('label shape:', train_label_batch.shape)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord = coord)

    try:

        for epoch in range(1, n_epochs + 1):

            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == 60 or epoch == 80:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)

            print("Training...")
            loss, acc = train_one_epoch(train_image_batch, train_label_batch)
            log_loss_accuracy(loss, acc, epoch, prefix='train')

            loss_all_epochs.append(loss)

            print("Validation...")
            loss, acc, nr, br, ir, kappa = test(test_image_batch, test_label_batch)
            log_loss_accuracy(loss, acc, epoch, prefix='valid')

            acc_all_epochs.append(acc)

            nr_all_epochs.append(nr)
            br_all_epochs.append(br)
            ir_all_epochs.append(ir)
            kappa_all_epochs.append(kappa)


            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if epoch >= 60 and epoch % 10 == 0: #
                save_model(global_step = epoch)

            if acc > best_acc:
                best_acc = acc
                save_model(global_step = 1)

        dataframe = pd.DataFrame({'train_loss': loss_all_epochs, 'accuracy': acc_all_epochs, 
            'normal_recall': nr_all_epochs, 'bleed_recall': br_all_epochs, 
            'inflam_recall': ir_all_epochs, 'kappa': kappa_all_epochs,})

        dataframe.to_csv("./acc_results/split1_repeat.csv", index = True, sep = ',')

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(seconds=total_training_time)))

    except tf.errors.OutOfRangeError:
        print("done!")
    finally:
        coord.request_stop()
        coord.join(threads)

def train_one_epoch(train_image_batch, train_label_batch):
    total_loss = []
    total_accuracy = []

    for i in range(num_train // batch_size):

        images, labels = sess.run([train_image_batch, train_label_batch])

        feed_dict = {
            x: images,
            y: labels,
            keep_prob: dropout_rate,
            is_training: True,
        }

        fetches = [train_op, net_loss, accuracy]

        result = sess.run(fetches, feed_dict=feed_dict)
        _, loss, acc = result

        total_loss.append(loss)
        total_accuracy.append(acc)

    mean_loss = np.mean(total_loss)
    mean_accuracy = np.mean(total_accuracy)

    return mean_loss, mean_accuracy


def test(test_image_batch, test_label_batch):

    input_rgb = "./visualization/input/"
    att_path = "./visualization/att_maps/"

    total_pred = []
    total_labels = []

    total_loss = []

    for i in range(num_test // batch_size):
        
        test_images, test_labels = sess.run([test_image_batch, test_label_batch])
        class_labels = np.argmax(test_labels, axis = 1).astype(np.int32)

        feed_dict = {
            x: test_images,
            y: test_labels,
            keep_prob: 1.0,
            is_training: False,
        }

        pred, att_map, loss, acc = sess.run([prediction, att_maps, net_loss, accuracy], feed_dict=feed_dict)

        pred = np.argmax(pred, axis = 1)
        labels = np.argmax(test_labels, axis = 1)

        total_pred.append(pred)
        total_labels.append(labels)
        total_loss.append(loss)

        for index in range(batch_size):
          img_index = i * batch_size + index
          save_img(test_images[index], img_index, input_rgb, img_name = '.jpg', mode = "image")
          save_img(att_map[index], img_index, att_path, img_name = '.jpg', mode = "heatmap")

    mean_loss = np.mean(total_loss)

    overall_acc, normal_recall, bleed_recall, inflam_recall, kappa = get_accuracy(preds = total_pred, labels = total_labels)

    false_index = np.where(np.equal(np.reshape(total_pred, (-1)), np.reshape(total_labels, (-1))) == False)[0]

    # print('====================show misclassified images==================')
    # print('the number of misclassified examples in Net1 is:', len(false_index1))
    # print false_index1

    return mean_loss, overall_acc, normal_recall, bleed_recall, inflam_recall, kappa     

# load_model()
train_all_epochs(learning_rate)   