import os
import cv2
import time
import shutil
from ops import *
from skimage import io
from datetime import timedelta

import numpy as np
import tensorflow as tf
import  pandas as pd # used to write and read csv files.
from skimage import io, data, color
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import data_provider
from data_provider import get_image_label_batch

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class DenseNet:
    def __init__(self, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=0.5,
                 bc_mode=True,
                 **kwargs):

        self.data_shape = (224, 224, 3)
        self.n_classes = 3
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = [2, 4, 8, 6]
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "totally %d layers." % (
                      model_type, self.total_blocks, self.depth))
        if bc_mode:
            # the layers in each block is consisted of bottleneck layers and composite layers,
            # so the number of composite layers should be half the total number.
            print("Build %s model with %d blocks, "
                  "totally %d layers." % (
                      model_type, self.total_blocks, self.depth))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        # self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0
        self.batch_size = 8

        self.num_train = 9688
        self.num_test = 2400
        # self.num_train = 8
        # self.num_test = 8

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    # if the save_path exists, use the save path
    # else create a save path
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}".format(
            self.model_type, self.growth_rate, self.depth)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss1, loss2, accuracy1, accuracy2, accuracy, 
                          epoch, prefix, should_print=True):
        if should_print:
            print("mean cross_entropy1: %f, mean accuracy1: %f" % (loss1, accuracy1))
            print('')
            print("mean cross_entropy2: %f, mean accuracy2: %f" % (loss2, accuracy2))
            print('')
            print("mean accuracy: %f" % (accuracy))

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='loss1_%s' % prefix, simple_value=float(loss1)),
            tf.Summary.Value(tag='loss2_%s' % prefix, simple_value=float(loss2)),
            tf.Summary.Value(tag='accuracy1_%s' % prefix, simple_value=float(accuracy1)),
            tf.Summary.Value(tag='accuracy2_%s' % prefix, simple_value=float(accuracy2)),
            tf.Summary.Value(tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])

        self.summary_writer.add_summary(summary, epoch)


    def _define_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')

        self.labels = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

        self.w1 = tf.placeholder(tf.float32)
        self.w2 = tf.placeholder(tf.float32)


    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        # the function batch_norm, conv2d, dropout are defined in the following part.
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)

            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, block, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block[block]):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and then average
        pooling
        """
        # call composite function with 1x1 kernel
        # reduce the number of feature maps by compression
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    # after block4, convert the 7*7 feature map to 1*1 by average pooling.
    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        spatial_features = output

        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        print(output.shape)
        # FC

        features = tf.reshape(output, [self.batch_size, -1]) 

        with tf.variable_scope("final_layer") as scope:
            output = self.conv2d(output, out_features = 3, kernel_size = 1)

            scope.reuse_variables()
            spatial_output = self.conv2d(spatial_features, out_features = 3, kernel_size = 1)
            spatial_pred = tf.nn.softmax(spatial_output) # (16, 14, 14, 3)
            # spatial_pred = tf.reshape(spatial_pred, [self.batch_size, -1, 3]) # (16, 196, 3)
        print output.shape, spatial_pred.shape

        logits = tf.reshape(output, [-1, self.n_classes])
        print features.shape, logits.shape

        return features, logits, spatial_pred


    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output


    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        # output = tf.contrib.layers.batch_norm(
        #     _input, scale=True, is_training=self.is_training,
        #     updates_collections=None)
        output = tf.contrib.layers.batch_norm(
            _input, decay = 0.9, epsilon = 1e-05, 
            center = True, scale=True, is_training=self.is_training,
            updates_collections=None)

        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output


    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())
            # an initializer that generates tensors with unit variance.

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)


    def max_unpool(self, pool, ind, output_shape, batch_size, name=None):

        with tf.variable_scope(name):
            pool_ = tf.reshape(pool, [-1])
            batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), [tf.shape(pool)[0], 1, 1, 1])
            # print ind, batch_range
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, [-1, 1])
            ind_ = tf.reshape(ind, [-1, 1])
            ind_ = tf.concat([b, ind_], 1)
            ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
            unpooled = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
            print "===========unpooled_output===========", unpooled.shape
            return unpooled

    # the inverse computation of average pooling
    def avg_unpool(self, _input, input_shape):
        # shape = _input.shape
        inference = tf.image.resize_nearest_neighbor(_input, 
                    size = [int(int(input_shape[1])*2), int(int(input_shape[2])*2)])
        return inference


    def deconv_block(self, _input, kernel_size, out_features, unpool = 'True', 
                     strides = [1,1,1,1], padding='SAME'):
        input_shape = _input.shape
        print "===========deconv_input===========", input_shape
        if unpool == 'True':
            _input = self.avg_unpool(_input,input_shape)
        un_relu = tf.nn.relu(_input)
        print "===========after_relu===========", un_relu.shape
        in_features = int(input_shape[-1])
        filter = self.weight_variable_msra(
            [kernel_size, kernel_size, out_features, in_features], name = 'deconv_kernel')
        output_shape = [int(input_shape[0]), int(int(input_shape[1])*2), 
                        int(int(input_shape[2])*2), out_features]
        tf.convert_to_tensor(output_shape)
        deconv = tf.nn.conv2d_transpose(un_relu, filter, output_shape, strides, padding, data_format = 'NHWC')
        print "===========deconv_output===========", deconv.shape
        return deconv


    def compute_saliency(self, f_maps):

        f_maps = tf.nn.relu(f_maps)
        s_map = tf.reduce_mean(f_maps, axis = -1, keepdims = True)

        s_map_min = tf.reduce_min(s_map, axis = [1, 2, 3], keepdims = True) # [batch_size, 8, 8, 1]
        s_map_max = tf.reduce_max(s_map, axis = [1, 2, 3], keepdims = True)
        s_map = tf.div(s_map - s_map_min, s_map_max - s_map_min) # [batch_size, 8, 8, 1]

        saliency_map = tf.tile(s_map, (1,1,1,3))
        saliency_map = tf.image.resize_images(saliency_map, (224, 224)) # (8, 224, 224, 3)

        return saliency_map


    def network(self, _input, ind = 1):
        """
        Define the structure of the backbone network, 
        this network will be used in both two branches.
        Input: image,
        Return: the feature vector of input image(batch_size, channels), 
        predicted logits(batch_size, 3), predicted spatial probability(batch_size, h, w, 3).
        """
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block

        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                _input,
                out_features=self.first_output_features,
                kernel_size=7, strides = [1, 2, 2, 1])
            print(output.shape)

        with tf.variable_scope("Initial_pooling"):
            if ind == 1:
                output, self.max_ind = tf.nn.max_pool_with_argmax(
                    output, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
            else:                
                output = tf.nn.max_pool(output, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
            print(output.shape)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(block, output, growth_rate, layers_per_block)
            print(output.shape)

            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)
                    print(output.shape)

        f_maps = output
        # the last block is followed by a "transition_to_classes" layer.
        with tf.variable_scope("Transition_to_classes"):
            features, logits, spatial_pred = self.transition_layer_to_classes(f_maps)

        return f_maps, logits


    def DGN_module(self, _input):
        # get the deconv saliency map through deconvolution
        with tf.variable_scope("deconv1"):
            deconv1 = self.deconv_block(_input, 3, 256)

        with tf.variable_scope("deconv2"):
            deconv2 = self.deconv_block(deconv1, 3, 128)

        with tf.variable_scope("deconv3"):
            deconv3 = self.deconv_block(deconv2, 3, 48)

        # with tf.variable_scope("unpool"):
        unpool1 = self.max_unpool(deconv3, self.max_ind, [None, 112, 112, 48], self.batch_size, "unpool")

        with tf.variable_scope("deconv4"):
            deconv4 = self.deconv_block(unpool1, kernel_size = 7, out_features = 24, unpool = 'False', strides = [1,2,2,1])

        deconv_f = deconv4

        return deconv_f


    def _build_graph(self):
              
        with tf.variable_scope("net1") as scope:
            self.input1 = self.images
            f_maps1, logits1 = self.network(self.input1, ind = 1)
            self.pred1 = tf.nn.softmax(logits1)
            self.avg_saliency = self.compute_saliency(f_maps1)

            deconv_feature = self.DGN_module(f_maps1)
            self.s_map1 = self.compute_saliency(deconv_feature)
            self.input2 = tf.concat([self.images, tf.reduce_mean(self.s_map1, axis = -1, keepdims = True)], axis = -1)


        with tf.variable_scope("net2") as scope:
            f_maps2, logits2 = self.network(self.input2, ind = 2)
            self.pred2 = tf.nn.softmax(logits2)

            # Saliency of the original features of block4 and the SACA feature.
            self.s_map2 = self.compute_saliency(f_maps2)

        self.cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits1, labels=self.labels))

        self.cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits2, labels=self.labels))


        self.grads = tf.gradients(self.cross_entropy2, self.images)[0]
        print('gradients from loss2 to input1:', self.grads)


        # regularize the variables that needs to be trained in Net1 or Net2.
        var_list = [var for var in tf.trainable_variables()]
        var_list1 = [var for var in tf.trainable_variables() if var.name.split('/')[0] == 'net1']
        var_list2 = [var for var in tf.trainable_variables() if var.name.split('/')[0] == 'net2']


        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        l2_loss1 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('/')[0] == 'net1'])
        l2_loss2 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('/')[0] == 'net2'])


        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)

        self.train_step = optimizer.minimize(
            self.cross_entropy1 + self.cross_entropy2 + l2_loss * self.weight_decay, var_list = var_list)


        correct_prediction1 = tf.equal(
            tf.argmax(self.pred1, 1),
            tf.argmax(self.labels, 1))
        self.correct_prediction1 = correct_prediction1
        self.accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

        correct_prediction2 = tf.equal(
            tf.argmax(self.pred2, 1),
            tf.argmax(self.labels, 1))
        self.correct_prediction2 = correct_prediction2
        self.accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

        self.final_pred = self.pred1 * self.w1 + self.pred2 * self.w2
        self.correct_prediction = tf.equal(
            tf.argmax(self.final_pred, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))        


    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        # self.batch_size = batch_size
        # reduce the lr at epoch1 and epoch2.
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        reduce_lr_epoch_3 = train_params['reduce_lr_epoch_3']
        reduce_lr_epoch_4 = train_params['reduce_lr_epoch_4']

        total_start_time = time.time()

        weight_1 = 0.5
        weight_2 = 0.5

        loss1_all_epochs = []
        loss2_all_epochs = []

        acc1_all_epochs = []
        acc2_all_epochs = []
        acc_all_epochs = []

        nr1_all_epochs = []
        br1_all_epochs = []
        ir1_all_epochs = []
        kappa1_all_epochs = []

        nr2_all_epochs = []
        br2_all_epochs = []
        ir2_all_epochs = []
        kappa2_all_epochs = []

        nr_all_epochs = []
        br_all_epochs = []
        ir_all_epochs = []
        kappa_all_epochs = []

        best_acc2 = 0.0

        self.train_image_batch, self.train_label_batch = \
                                    get_image_label_batch(batch_size, shuffle=True, name='train')
        self.test_image_batch, self.test_label_batch = \
                                    get_image_label_batch(batch_size, shuffle=False, name='4aug_test')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord = coord)

        try:

            for epoch in range(1, n_epochs + 1):
                print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
                start_time = time.time()
                if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2 \
                or epoch == reduce_lr_epoch_3 or epoch == reduce_lr_epoch_4:
                    learning_rate = learning_rate / 10
                    print("Decrease learning rate, new lr = %f" % learning_rate)

                print("Training...")
                loss1, loss2, acc1, acc2 = self.train_one_epoch(batch_size, learning_rate)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss1, loss2, acc1, acc2, acc2, epoch, prefix='train')

                loss1_all_epochs.append(loss1)
                loss2_all_epochs.append(loss2)

                if train_params.get('validation_set', False):
                    print("Validation...")
                    loss1, loss2, acc1, acc2, acc, nr1, br1, ir1, kappa1, nr2, br2, ir2, \
                        kappa2, nr, br, ir, kappa = self.test(weight_1, weight_2, batch_size)

                    if self.should_save_logs:
                        self.log_loss_accuracy(loss1, loss2, acc1, acc2, acc, epoch, prefix='valid')

                    error1 = 1 - acc1
                    error2 = 1 - acc2

                    epsilon = 1e-8
                    weight1 = np.log((1 - error1)/(error1 + epsilon))/2
                    weight2 = np.log((1 - error2)/(error2 + epsilon))/2
                    weight_1 = np.exp(weight1)/(np.exp(weight1) + np.exp(weight2))
                    weight_2 = np.exp(weight2)/(np.exp(weight1) + np.exp(weight2))

                    # self.weight1 = weight_1
                    # self.weight2 = weight_2

                    print "========weight 1=======", weight_1, "=======weight 2=======", weight_2


                    acc1_all_epochs.append(acc1)
                    acc2_all_epochs.append(acc2)
                    acc_all_epochs.append(acc)

                    nr1_all_epochs.append(nr1)
                    br1_all_epochs.append(br1)
                    ir1_all_epochs.append(ir1)
                    kappa1_all_epochs.append(kappa1)

                    nr2_all_epochs.append(nr2)
                    br2_all_epochs.append(br2)
                    ir2_all_epochs.append(ir2)
                    kappa2_all_epochs.append(kappa2)

                    nr_all_epochs.append(nr)
                    br_all_epochs.append(br)
                    ir_all_epochs.append(ir)
                    kappa_all_epochs.append(kappa)


                time_per_epoch = time.time() - start_time
                seconds_left = int((n_epochs - epoch) * time_per_epoch)
                print("Time per epoch: %s, Est. complete in: %s" % (
                    str(timedelta(seconds=time_per_epoch)),
                    str(timedelta(seconds=seconds_left))))

                if self.should_save_model:
                    if epoch >= 60 and epoch % 5 == 0: #
                        self.save_model(global_step = epoch)

                    if acc2 > best_acc2:
                        best_acc2 = acc2
                        self.save_model(global_step = 1)


            dataframe = pd.DataFrame({'train_loss1': loss1_all_epochs, 'train_loss2': loss2_all_epochs,'accuracy1': acc1_all_epochs, 
                'accuracy2': acc2_all_epochs, 'accuracy': acc_all_epochs, 'normal_recall_1': nr1_all_epochs, 'normal_recall_2': nr2_all_epochs,
                'normal_recall': nr_all_epochs, 'bleed_recall_1': br1_all_epochs, 'bleed_recall_2': br2_all_epochs, 'bleed_recall': br_all_epochs, 
                'inflam_recall_1': ir1_all_epochs, 'inflam_recall_2': ir2_all_epochs, 'inflam_recall': ir_all_epochs, 
                'kappa1': kappa1_all_epochs, 'kappa2': kappa2_all_epochs, 'kappa': kappa_all_epochs,})

            dataframe.to_csv("./acc_results/result.csv", index = True, sep = ',')

            total_training_time = time.time() - total_start_time
            print("\nTotal training time: %s" % str(timedelta(
                seconds=total_training_time)))

        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
            coord.join(threads)

        return weight_1, weight_2

    def train_one_epoch(self, batch_size, learning_rate):

        total_loss1 = []
        total_loss2 = []

        total_accuracy1 = []
        total_accuracy2 = []

        for i in range(self.num_train // batch_size):
            images, labels = self.sess.run([self.train_image_batch, self.train_label_batch])

            # the class_labels for features in Net1 are 0,1,2
            class_labels1 = np.argmax(labels, axis = 1).astype(np.int32)
            # the class_labels for features in Net2 are 3,4,5
            class_labels2 = class_labels1 + 3

            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }


            fetches = [self.train_step, self.cross_entropy1, self.cross_entropy2, self.accuracy1, self.accuracy2]

            results = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss1, loss2, acc1, acc2 = results
            
            # print(pred)
            total_loss1.append(loss1)
            total_loss2.append(loss2)

            total_accuracy1.append(acc1)
            total_accuracy2.append(acc2)

            if self.should_save_logs:
                self.batches_step += 1
                # save loss and accuracy into Summary
                self.log_loss_accuracy(
                    loss1, loss2, acc1, acc2, acc2, self.batches_step, 
                    prefix='per_batch', should_print=False)

        mean_loss1 = np.mean(total_loss1)
        mean_loss2 = np.mean(total_loss2)

        mean_accuracy1 = np.mean(total_accuracy1)
        mean_accuracy2 = np.mean(total_accuracy2)

        return mean_loss1, mean_loss2, mean_accuracy1, mean_accuracy2


    def test(self, w1, w2, batch_size):

        input_path = "./visualization/input/"
        saliency_path = "./visualization/att_maps/"

        total_loss1 = []
        total_loss2 = []

        total_pred1 = []
        total_pred2 = []
        total_pred = []
        total_labels = []

        epsilon = 1e-8

        for i in range(self.num_test // batch_size):
            test_images, test_labels = self.sess.run([self.test_image_batch, self.test_label_batch])

            feed_dict = {
                self.images: test_images,
                self.labels: test_labels,
                self.is_training: False,
                self.w1: w1,
                self.w2: w2,
            }
    

            fetches = [self.cross_entropy1, self.cross_entropy2, self.avg_saliency, \
                self.s_map1, self.s_map2, self.pred1, self.pred2, self.final_pred]

            loss1, loss2, avg_s1, s_map1, s_map2, pred1, pred2, pred = self.sess.run(fetches, feed_dict=feed_dict)


            total_loss1.append(loss1)
            total_loss2.append(loss2)

            pred1 = np.argmax(pred1, axis = 1)
            pred2 = np.argmax(pred2, axis = 1)
            pred = np.argmax(pred, axis = 1)
            labels = np.argmax(test_labels, axis = 1)

            total_pred1.append(pred1)
            total_pred2.append(pred2)
            total_pred.append(pred)
            total_labels.append(labels)


            for index in range(batch_size):
                img_index = i * batch_size + index
                save_img(test_images[index], img_index, input_path, img_name = '.jpg', mode = "image")
                save_img(s_map1[index], img_index, saliency_path, img_name = '_saliency.jpg', mode = "heatmap")
                save_img(avg_s1[index], img_index, saliency_path, img_name = '_avg_saliency.jpg', mode = "heatmap")
                # plt.figure()
                # plt.imshow(avg_s1[index])
                # plt.show()


        # print pred1, pred2, labels
        mean_loss1 = np.mean(total_loss1)
        mean_loss2 = np.mean(total_loss2)

        overall_acc_1, normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1 = get_accuracy(preds = total_pred1, labels = total_labels)
        overall_acc_2, normal_recall_2, bleed_recall_2, inflam_recall_2, kappa_2 = get_accuracy(preds = total_pred2, labels = total_labels)
        overall_acc, normal_recall, bleed_recall, inflam_recall, kappa = get_accuracy(preds = total_pred, labels = total_labels)

        # print ("========================Network1======================")
        # print_scores(normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1)

        # print ("========================Network2======================")
        # print_scores(normal_recall_2, bleed_recall_2, inflam_recall_2, kappa_2)

        # print ("=================Ensembled performance================")
        # print_scores(normal_recall, bleed_recall, inflam_recall, kappa)


        false_index1 = np.where(np.equal(np.reshape(total_pred1, (-1)), np.reshape(total_labels, (-1))) == False)[0]
        false_index2 = np.where(np.equal(np.reshape(total_pred2, (-1)), np.reshape(total_labels, (-1))) == False)[0]
        false_index = np.where(np.equal(np.reshape(total_pred, (-1)), np.reshape(total_labels, (-1))) == False)[0]

        # print('====================show misclassified images==================')
        # print('the number of misclassified examples in Net1 is:', len(false_index1))
        # print false_index1

        # print('the number of misclassified examples in Net2 is:', len(false_index2))
        # print false_index2

        return mean_loss1, mean_loss2, overall_acc_1, overall_acc_2, overall_acc, \
            normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1, normal_recall_2, bleed_recall_2, \
            inflam_recall_2, kappa_2, normal_recall, bleed_recall, inflam_recall, kappa