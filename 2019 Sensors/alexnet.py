import tensorflow as tf
import numpy as np
from ops import *

class AlexNet(object):
  
  def __init__(self, x, keep_prob, num_classes, 
               weights_path = 'DEFAULT'):
    
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    
    # Call the network function to build the computational graph of AlexNet
    self.network()
    
  def network(self):
    
    # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
    conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
    norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
    
    # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
    conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 1, name = 'conv2')
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
    norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
    
    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')
    
    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 1, name = 'conv4')
    
    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 1, name = 'conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')
    self.att_map = make_img(pool5, dst_size = 227)
    
    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6*6*256])
    fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
    dropout6 = dropout(fc6, self.KEEP_PROB)
    
    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
    dropout7 = dropout(fc7, self.KEEP_PROB)
    
    # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')
     
  
"""
Predefine all necessary layer for the AlexNet
""" 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    # weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters], 
    #   initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01), trainable = True)
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters], 
      initializer = tf.contrib.layers.variance_scaling_initializer(), trainable = True)
    # weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters], trainable = True)
    biases = tf.get_variable('biases', shape = [num_filters], initializer = tf.zeros_initializer(), trainable = True)  
    
    
    if groups == 1:
      conv = convolve(x, weights)            
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())    
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)
        
    return relu
  
def fc(x, num_in, num_out, name, relu = True):

  with tf.variable_scope(name) as scope:    
    # Create tf variables for the weights and biases
    # weights = tf.get_variable('weights', shape=[num_in, num_out], 
    #   initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01), trainable=True)
    weights = tf.get_variable('weights', shape=[num_in, num_out], 
      initializer = tf.contrib.layers.variance_scaling_initializer(), trainable=True)
    # weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)    
    biases = tf.get_variable('biases', [num_out], initializer = tf.zeros_initializer(), trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)
  
def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
  
def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)
  
    