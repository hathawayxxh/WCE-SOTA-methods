import tensorflow as tf
import tensorflow.contrib as tf_contrib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

##########################################################
### Net measure
##########################################################
def net_measure(pred, labels):
    correct_pred = tf.equal(tf.argmax(pred, 1),tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return correct_pred, accuracy

def get_accuracy(preds, labels):
    """
    Overall accuracy
    """
    preds = np.reshape(preds, (-1))
    labels = np.reshape(labels, (-1))
    accuracy = round(accuracy_score(y_true = labels, y_pred = preds), 5)
    """
    Per_class_recall
    """
    matrix = confusion_matrix(y_true = labels, y_pred = preds)
    print ("confusion_matrix:", matrix)
    recalls = matrix.diagonal().astype('float')/matrix.sum(axis = 1)

    normal_recall = round(recalls[0], 5)
    inflam_recall = round(recalls[1], 5)
    bleed_recall = round(recalls[2], 5)

    """
    Cohen kappa
    """  
    kappa = round(cohen_kappa_score(y1 = preds, y2 = labels), 5)

    return accuracy, normal_recall, bleed_recall, inflam_recall, kappa


####################################################
### Regular operations
####################################################
def make_img(_input, dst_size = 128):
    """
    Normalize the given input and resize it into an image with size [128, 128, 3]
    """
    x = tf.nn.relu(_input)

    if int(x.get_shape()[-1])!= 1:
        x = tf.reduce_mean(x, axis = -1, keepdims = True)

    x_max = tf.reduce_max(x, axis = [1, 2, 3], keepdims = True)
    x_min = tf.reduce_min(x, axis = [1, 2, 3], keepdims = True)

    x_norm = tf.div(x - x_min, x_max - x_min)
    output = tf.image.resize_images(tf.tile(x_norm, (1,1,1,3)), (dst_size, dst_size))

    return output


def save_img(img, img_index, root_path, img_name, mode = "image"):
    img = np.uint8(255 * img)
    if mode == "image":            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif mode == "heatmap":
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img_path = os.path.join(root_path, str(img_index) + img_name)
    cv2.imwrite(img_path, img)