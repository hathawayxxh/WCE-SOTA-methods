import time
from ops import *
from utils import *
import utils
import os
import sys
import os.path as osp
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
import math
from tensorflow.python.ops import array_ops

weight_decay = 0.00001
label_count = 3
batch_size = 8
growth_k=12
SN=True 

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("tfdata_path", './tfrecord/', "tf-records save path")
flags.DEFINE_integer("batch_size", batch_size, "batch size [16]")
flags.DEFINE_integer("min_after_dequeue", 100, "min nums data filename in queue")
flags.DEFINE_integer("capacity", 200, "capacity")
flags.DEFINE_string("preprocessing_name", 'default', "pre-processing_name")
flags.DEFINE_integer("train_image_size", 128, "train_image_size")
config = flags.FLAGS
# FLAGS._parse_flags()
FLAGS(sys.argv)

w_init = tf.contrib.layers.xavier_initializer(uniform=False)    
def mkdir_if_missing(d):
    if not osp.isdir(d):
        os.makedirs(d)
        
def conv2pool(input, filters, kernel, decay, stride, scope, training, reuse=True):
    x = conv(input, channels=filters, kernel=kernel, stride=stride, pad=1, sn=SN, use_bias=False, scope=scope)
    x = batch_norm(x, reuse, is_training = training, scope=scope + '_batch1')
    x = tf.nn.relu(x)
    x = tf.contrib.layers.max_pool2d(inputs=x, kernel_size=[2, 2], stride=2, padding='VALID')
    return x

def bottleneck_layer_2d(input, filters, drop_rate, decay, scope, training, reuse):
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        x = batch_norm(input, reuse, is_training = training, scope=scope + '_batch1')
        x = tf.nn.relu(x)
        x = conv(x, channels=4 * filters, kernel=1,stride=1,pad=0,sn=SN, use_bias=False, scope=scope+'_conv1')
        #x = tf.contrib.layers.dropout(inputs=x, keep_prob=drop_rate, is_training=training)
        
        x = batch_norm(x, reuse, is_training = training, scope=scope + '_batch2')
        x = tf.nn.relu(x)
        x = conv(x, channels=filters, kernel=3,stride=1,pad=1,sn=SN, use_bias=False, scope=scope+'_conv2')
        #x = tf.contrib.layers.dropout(inputs=x, keep_prob=drop_rate, is_training=training)
        return x

def transition_layer_2d(input, filters, drop_rate, decay, scope, training, reuse):
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        x = batch_norm(input, reuse, is_training = training, scope=scope + '_batch')
        x = tf.nn.relu(x)
        x = conv(x, channels=filters, kernel=1,stride=1,pad=0,sn=SN, use_bias=False, scope=scope+'_conv')
        #x = tf.contrib.layers.dropout(x, keep_prob=drop_rate, is_training=training)
        x = tf.contrib.layers.avg_pool2d(inputs=x, kernel_size=[2, 2], stride=2, padding='VALID')
        return x

def dense_block_2d(input, filters, nb_layers, drop_rate, decay, training, reuse, scope):
    with tf.name_scope(scope):
        layers_concat = list()
        layers_concat.append(input)
        x = input
        for i in range(nb_layers):
            x = bottleneck_layer_2d(x, filters, drop_rate, decay, training=training,
                                    reuse=reuse, scope=scope + '_bottleN_' + str(i+1))
            
            ####with ADB
            gamma = tf.get_variable(scope+"gamma"+str(i), [1], initializer=tf.constant_initializer(1.0))
            layers_concat.append(gamma*x)
            
            ####without ADB
            #layers_concat.append(x)
            x = tf.concat(layers_concat, axis=-1)
        return x


def AAM(x, channels, de=4, scope='AAM', trainable=True, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):        
        f, offset1 = deform_con2v(x, num_outputs=channels // de, kernel_size=3, stride=1, trainable=trainable,  name=scope+'f_conv', reuse=reuse)
        g, offset2 = deform_con2v(x, num_outputs=channels // de, kernel_size=3, stride=1, trainable=trainable,  name=scope+'g_conv', reuse=reuse)
        h = conv(x, channels, kernel=1, stride=1, sn=SN, scope='h_conv')

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta_a = tf.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta_a, hw_flatten(h)) # [bs, N, C]
        
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
        att = gamma * o
        x = att + x

    return f, g, h, x, gamma


def DTOA(x, channels, de=4, scope='DTOA', trainable=True, reuse=False):
    """
    A combination of AAM and second-order non-local (SONA).
    It is completed by replacing the theta: 1*1 conv in SONA by deformable convolution.
    """
    with tf.variable_scope(scope, reuse=reuse):  

        c_num = channels // de

        f, offset1 = deform_con2v(
            x, num_outputs=c_num, kernel_size=3, stride=1, trainable=trainable,  name=scope+'f_conv', reuse=reuse)
        f = tf.nn.relu(f) #[bs, h, w, c_num]

        h = conv(x, channels, kernel=1, stride=1, sn=SN, scope='h_conv') #[bs, h, w, c_num]

        # N = h * w
        ### compute the spatial correlation.        
        I_hat = (-1./c_num/c_num) * tf.ones([c_num, c_num]) + (1./c_num) * tf.eye(c_num)
        I_hat = tf.tile(tf.expand_dims(I_hat, 0), [x.shape[0], 1, 1]) # # (8, c, c)
        correlation = tf.matmul(tf.matmul(hw_flatten(f), I_hat), hw_flatten(f), False, True)
        correlation1 = tf.nn.softmax(correlation, axis = -1)
        correlation = tf.nn.softmax(correlation/(tf.sqrt(tf.cast(c_num, tf.float32))), axis = -1) # (8, hw, hw)
        # correlation = tf.nn.softmax(correlation, axis = -1)

        ### aggregated feature
        o = tf.matmul(correlation, hw_flatten(h)) # [bs, N, C]
        
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
        output = gamma * o + x

    return f, o, output, correlation


# def DTOA(x, channels, de=4, scope='DTOA', trainable=True, reuse=False):
#     """
#     A combination of AAM and second-order non-local (SONA).
#     It is completed by replacing the theta: 1*1 conv in SONA by deformable convolution.
#     """
#     with tf.variable_scope(scope, reuse=reuse):  

#         # c_num = channels // de
#         c_num = channels

#         f, offset1 = deform_con2v(
#             x, num_outputs=c_num, kernel_size=3, stride=1, trainable=trainable,  name=scope+'f_conv', reuse=reuse)
#         # f = x
#         f = tf.nn.relu(f) #[bs, h, w, c_num]

#         h = conv(x, channels, kernel=1, stride=1, sn=SN, scope='h_conv') #[bs, h, w, c_num]

#         # N = h * w
#         ### compute the spatial correlation.        
#         I_hat = (-1./c_num/c_num) * tf.ones([c_num, c_num]) + (1./c_num) * tf.eye(c_num)
#         I_hat = tf.tile(tf.expand_dims(I_hat, 0), [x.shape[0], 1, 1]) # # (8, c, c)
#         correlation = tf.matmul(tf.matmul(hw_flatten(f), I_hat), hw_flatten(f), False, True)
#         # correlation = correlation/(tf.sqrt(tf.cast(c_num, tf.float32)))
#         correlation = tf.nn.softmax(correlation/(tf.sqrt(tf.cast(c_num, tf.float32))), axis = -1) # (8, hw, hw)
#         # correlation = tf.nn.softmax(20 * correlation, axis = -1)

#         ### aggregated feature
#         o = tf.matmul(correlation, hw_flatten(h)) # [bs, N, C]
        
#         # gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

#         o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
#         # o = conv(o, channels, kernel=1, stride=1, sn=SN, scope='final_conv')
#         output = o + x

#     return f, o, output, correlation



def nonlocal_block(x, channels, de=4, scope='Nonlocal', trainable=True, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):        

        f = conv(x, channels // de, kernel=1, stride=1, sn=SN, scope='f_conv')
        g = conv(x, channels // de, kernel=1, stride=1, sn=SN, scope='g_conv')
        h = conv(x, channels, kernel=1, stride=1, sn=SN, scope='h_conv')

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta_a = tf.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta_a, hw_flatten(h)) # [bs, N, C]
        
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
        att = gamma * o
        x = att + x

    return f, g, h, x


def Squeeze_excitation_layer(input, out_dim, ratio, scope='squeeze_excitation', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) :
        squeeze = global_avg_pool(input)

        excitation = tf.layers.dense(squeeze, units=out_dim / ratio, name='fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(excitation, units=out_dim, name='fully_connected2')
        excitation = tf.nn.softmax(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input * excitation

        return scale

def g_theta(arccos, k):
    sigmoid1 = (1+tf.exp(-math.pi/2.0/k))/(1-tf.exp(-math.pi/2.0/k))
    sigmoid2 = (1-tf.exp(arccos/k-math.pi/2.0/k))/(1+tf.exp(arccos/k-math.pi/2.0/k))
    cos_t = sigmoid1 * sigmoid2
    return cos_t

def ACLoss(embedding, labels, out_num, w_init=None, s=64., m=0.5, k = 0.3, is_training=True):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('ACLoss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        print(embedding, weights)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        arccos = tf.acos(tf.matmul(embedding, weights, name='cos_t'))
        cos_t = g_theta(arccos, k)
        pred = s*cos_t
        output = s*cos_t
        
        ######regularization term calculate##########
        A = [1] * out_num
        exclude = tf.to_float(tf.matrix_diag(A))
        zeros = array_ops.zeros_like(exclude, dtype=exclude.dtype)
        reg = tf.matmul(weights, weights, transpose_a=True)
        reg = tf.where(exclude>0.0, zeros, reg)
        regularization = tf.reduce_sum(reg) / ((out_num-1) * out_num)
        
        if is_training:
            arccos_mt = tf.acos(tf.matmul(embedding, weights, name='cos_t')) + m
            cos_mt =  s*g_theta(arccos_mt, k)

            # this condition controls the theta+m should in range [0, pi]
            #      0<=theta+m<=pi
            #     -m<=theta<=pi-m
            cond_v = cos_t - threshold
            cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

            keep_val = s*(cos_t - mm)
            cos_mt_temp = tf.where(cond, cos_mt, keep_val)

            mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
            # mask = tf.squeeze(mask, 1)
            inv_mask = tf.subtract(1., mask, name='inverse_mask')

            s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

            output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='ACLoss_output')
    return pred, output, reg, regularization

def triple_anet(image, labels, label_count, drop_rate=1.0, decay=0.9, growth_k=12, trainable=True, reuse=False, scope='dis'):
    end_points ={}        
    with tf.variable_scope(scope, reuse=reuse):
        print('model_name:triple_anet')
        #################################################################################
        ###conv pool  input: 128  output: 64
        #################################################################################
        logits = conv2pool(image, filters=2*growth_k, kernel=3, stride=1, decay=decay,
                      training=trainable, reuse=reuse, scope='conv2pool_1')
        print(logits)
            
        #################################################################################
        ###dense_block1  &&  trans_layer1  input: 64  output: 32
        #################################################################################
        logits = dense_block_2d(logits, growth_k, nb_layers=6, drop_rate=drop_rate, decay=decay,
                                training=trainable, reuse=reuse, scope='dense_block_1')
        print(logits)
        logits = transition_layer_2d(logits, filters=0.5*int(logits.shape[-1]), drop_rate=drop_rate, decay=decay,
                                     training=trainable, reuse=reuse,
                                     scope='trans_layer_1')#96
        print(logits)

        # f0, g0, h0, logits, _ = AAM(logits, int(logits.shape[-1]), de=4, scope='AAM0', trainable=trainable, reuse=reuse)      
        # print(logits)
        # end_points['f0'] = make_png(f0, 4)          
        # end_points['g0'] = make_png(g0, 4)          
        # end_points['h0'] = make_png(tf.abs(h0), 4)
        # end_points['AAM0'] = make_png(logits, 4)    
 
        f0, g0, h0, logits = nonlocal_block(logits, int(logits.shape[-1]), de=4, scope='NL0', trainable=trainable, reuse=reuse) 
        print ("nonlocal output", logits)
        end_points['NL0'] = make_png(logits, 4)   


        # f0, h0, logits, _ = DTOA(logits, int(logits.shape[-1]), de=4, scope='DTOA0', trainable=trainable, reuse=reuse) 
        # print (logits)
        # end_points['f0'] = make_png(f0, 4)
        # end_points['h0'] = make_png(h0, 4)        
        # end_points['DTOA0'] = make_png(logits, 4)  

        #################################################################################
        ###dense_block2  &&  trans_layer2  input: 32  output: 16
        #################################################################################
        logits = dense_block_2d(logits, growth_k, nb_layers=12, drop_rate=drop_rate, decay=decay,training=trainable, reuse=reuse, scope='dense_block_2')
        print(logits)
        logits = transition_layer_2d(logits, filters=0.5*int(logits.shape[-1]),  drop_rate=drop_rate, decay=decay,
                                     training=trainable, reuse=reuse, scope='trans_layer_2')
        print(logits)
        end_points['att_b2'] = make_png(logits, 8)            

        # f1, g1, h1, logits, gamma = AAM(logits, int(logits.shape[-1]), de=4, scope='AAM1', trainable=trainable, reuse=reuse)
        # end_points['f1'] = make_png(tf.abs(f1), 8)          
        # end_points['g1'] = make_png(tf.abs(g1), 8)
        # end_points['h1'] = make_png(tf.abs(h1), 8)
        # end_points['AAM1'] = make_png(logits, 8)   
        # end_points['gamma'] = gamma         

        f1, g1, h1, logits = nonlocal_block(
            logits, int(logits.shape[-1]), de=4, scope='NL1', trainable=trainable, reuse=reuse) 
        print (logits)
        end_points['f1'] = make_png(tf.abs(f1), 8)          
        end_points['g1'] = make_png(tf.abs(g1), 8)
        end_points['h1'] = make_png(tf.abs(h1), 8)
        end_points['NL1'] = make_png(logits, 8)   


        # f1, h1, logits, correlation1 = DTOA(logits, int(logits.shape[-1]), de=4, scope='DTOA1', trainable=trainable, reuse=reuse) 
        # print (logits)
        # end_points['f1'] = make_png(f1, 4)
        # end_points['h1'] = make_png(h1, 4)        
        # end_points['DTOA1'] = make_png(logits, 8)
        # end_points['correlation1'] = correlation1  
            
        #################################################################################
        ###dense_block3  &&  trans_layer3  input: 16  output: 8
        #################################################################################
        logits = dense_block_2d(logits, growth_k, nb_layers=24,  drop_rate=drop_rate, decay=decay,training=trainable, reuse=reuse, scope='dense_block_3')
        print(logits)
        logits = transition_layer_2d(logits, filters=0.5*int(logits.shape[-1]),  drop_rate=drop_rate, decay=decay,
                                     training=trainable, reuse=reuse, scope='trans_layer_3')
        print(logits)

        end_points['att_b3'] = make_png(logits, 16)            

        #################################################################################
        ###dense_block4  input: 8  output: 8
        #################################################################################
        logits = dense_block_2d(logits, growth_k, nb_layers=16, decay=decay,  drop_rate=drop_rate,training=trainable, reuse=reuse, scope='dense_block_4')
        print(logits) 

        end_points['att_b4'] = make_png(logits, 16)            
        
        logits = global_avg_pool(logits, name='Global_avg_pooling_pool')
        feature = tf.squeeze(logits)
        print(feature)
        #### with AC Loss
        pred, ys_, reg, regularization = ACLoss(logits, labels, label_count, w_init=w_init, s=64., m=0.5, k = 0.3, is_training=trainable)
        #### without AC Loss, with softmax loss instead
        #ys_ = tf.layers.dense(inputs=logits, units=label_count, name='fc2')
        #pred = ys_
        #reg = 0
        #regularization = 0
        print(ys_)
        return feature, pred, ys_, end_points, reg, regularization
    
graph = tf.Graph()

xs = tf.placeholder(tf.float32, shape=[batch_size, 128, 128, 3], name='images')
ys = tf.placeholder(tf.int32, shape=[batch_size], name='label')
lr = tf.placeholder("float", shape=[])

test_image_batch, test_label_batch = utils.get_image_label_batch(config, shuffle=False, name='test')

####pred = cos theta, ys_ = used to calculate loss function
feature, pred, ys_, train_end_points, reg, regularization = triple_anet(
    xs, ys, label_count, drop_rate=1.0, decay=0.9, growth_k=growth_k, trainable=True, reuse=False)    
   
cross_entropy = class_loss(ys_, ys, label_count)
l2 = tf.add_n([weight_decay * tf.nn.l2_loss(var) for var in tf.trainable_variables()])
d_optim = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 + regularization)
correct_prediction = tf.equal(tf.cast(tf.argmax(pred, 1), tf.int32), ys)
train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""" validation """
val_feature, val_pred, val_ys_, val_end_points,_,_ = triple_anet(
    xs, ys, label_count, drop_rate=1, decay=0.9, growth_k=growth_k, trainable=False, reuse=True)  
val_pred = tf.cast(tf.argmax(val_pred, 1), tf.int32)
val_correct_prediction = tf.equal(val_pred, ys)
test_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))
        
""" Summary """
#class_sum = tf.summary.scalar("class_loss", cross_entropy)
#center_sum = tf.summary.scalar("center_loss", center_loss)
#train_eval_sum = tf.summary.scalar('train_accuracy', train_accuracy)
#test_eval_sum = tf.summary.scalar('test_accuracy', test_accuracy)
    
#os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
sess_config = tf.ConfigProto(gpu_options=gpu_options,
                         log_device_placement=False,
                         allow_soft_placement=False)

with tf.Session(config=sess_config) as sess:
    saver = tf.train.Saver(tf.global_variables())
    tf.global_variables_initializer().run()
    
    model_name = 'split2'
    load_fn = slim.assign_from_checkpoint_fn(os.path.join(
        './logs/'+model_name+'/model/', 'triple_anet'+'.model-60'),tf.global_variables(),ignore_missing_vars=True)
    load_fn(sess)
    print(model_name+' have been loaded')
        
    mkdir_if_missing('./logs/'+model_name+'/')
    mkdir_if_missing('./logs/'+model_name+'/result/')
    
    parameters = utils.count_trainable_params()
    print("Total training params: %.1fM \r\n" % (parameters / 1e6))

    infile = open('./logs/'+model_name+'/result.txt','w')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        GGamma = {}
        start_time = time.time()    
        learning_rate = 0.001
        batch_count = np.int32(9688 / batch_size)
        counter = 0
        counter1 = 0        
        for batch_idx in range(1):
            counter += 1
            if batch_idx == 0:
                test_acc = 0
                for i in range(np.int32(600 / batch_size)):
                    counter1 += 1
                    test_images, test_label = sess.run([test_image_batch, test_label_batch])
                    test_result, test_label, test_results, te_end_points = sess.run(
                        [val_pred, ys, test_accuracy, val_end_points],feed_dict = { xs: test_images, ys: test_label})
                    #test_results, summary_str, te_end_points = sess.run([test_accuracy, test_eval_sum, val_end_points],feed_dict = { xs: test_images, ys: test_label})
                    #writer.add_summary(summary_str, counter1)
                    #print("Epoch: [%3d] time: %4.4f, test_accuracy: %.8f" % (epoch, time.time() - start_time, test_results))
                    infile.write(str(test_result)+' '+str(test_label)+'\n')
                    infile.write(str(test_results)+'\n')
                    print(test_result, test_label)
                    print(test_results)
                            
                    test_acc = test_acc+test_results

                    # test_att0 = te_end_points['AAM0']
                    # att_AAM2 = te_end_points['AAM1']
                    # att_NL0 = te_end_points['NL0']
                    # att_NL1 = te_end_points['NL1']
                    f1 = te_end_points['f1']
                    # g1 = te_end_points['g1']
                    h1 = te_end_points['h1']
                    # att_DTOA0 = te_end_points['DTOA0']
                    att_DTOA1 = te_end_points['NL1']
                    att_b2 = te_end_points['att_b2']
                    att_b3 = te_end_points['att_b3']
                    att_b4 = te_end_points['att_b4']

                    # print ("gamma_AAM1", te_end_points['gamma'])
                    # print ("correlation1", te_end_points['correlation1'])

                    tot_num_samples = FLAGS.batch_size
                    m_h = int(np.floor(np.sqrt(tot_num_samples)))
                    m_w = int(np.floor(np.sqrt(tot_num_samples)))

                    # save_images(test_att0[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'test_AAM0.jpg')
                    # save_images(att_AAM2[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'att_AAM2.jpg')
                    # save_images(att_NL0[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'att_NL0.jpg')
                    # save_images(att_NL1[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'att_NL1.jpg')

                    save_images(f1[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'f1.jpg')
                    # save_images(g1[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'g1.jpg')
                    save_images(h1[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'h1.jpg')                    

                    # save_images(att_DTOA0[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'att_DTOA0.jpg')
                    # save_images(att_DTOA1[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'att_DTOA1.jpg')                    
                    save_images(att_b2[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'att_b2.jpg')
                    save_images(att_b3[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'att_b3.jpg')
                    save_images(att_b4[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'att_b4.jpg')
                    save_images(test_images[:m_h * m_w, :,:,:],[m_h, m_w], './logs/'+model_name+'/saliency/'+str(counter1)+'test.jpg')

                test_acc =test_acc/counter1
            
            #print("Epoch: [%3d] time: %4.4f, sum_test_accuracy: %.8f" % (epoch, time.time() - start_time, test_acc))
            infile.write(str(test_acc)+'\n')
            print(test_acc)
            infile.close()
            
    except tf.errors.OutOfRangeError:
        print("done!")
    finally:
        coord.request_stop()
        coord.join(threads)
