import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

import config_res as config
from common.cnn_utils_res import *

import common.resnet_rgb_model as model
import common.resnet_depth_model as model_depth
import tqdm

batch_size = config.net_params['batch_size']
current_epoch = config.net_params['load_epoch']
IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']


class Nets:
    def __init__(self, cam, velo, rgb_phase, depth_phase, fc_keep_prob):
        self.cam = cam
        self.velo = velo
        self.rgb_pahse = rgb_phase
        self.depth_phase = depth_phase
        self.fc_keep_prob = fc_keep_prob

    def NonLocalBlock(self, input_x, out_channels, sub_sample=True, is_bn=False, scope='NonLocalBlock'):
        batchsize, height, width, in_channels = input_x.get_shape().as_list()
        batchsize = tf.shape(input_x)[0]
        with tf.variable_scope(scope) as sc:
            with tf.variable_scope('g') as scope:
                g = slim.conv2d(input_x, out_channels, [1, 1], stride=1, scope='g')
                if sub_sample:
                    g = slim.max_pool2d(g, [2, 2], stride=2, scope='g_max_pool')

            with tf.variable_scope('phi') as scope:
                phi = slim.conv2d(input_x, out_channels, [1, 1], stride=1, scope='phi')
                if sub_sample:
                    phi = slim.max_pool2d(phi, [2, 2], stride=2, scope='phi_max_pool')

            with tf.variable_scope('theta') as scope:
                theta = slim.conv2d(input_x, out_channels, [1, 1], stride=1, scope='theta')

            g_x = tf.reshape(g, [batchsize, out_channels, -1])
            g_x = tf.transpose(g_x, [0, 2, 1])

            theta_x = tf.reshape(theta, [batchsize, out_channels, -1])
            theta_x = tf.transpose(theta_x, [0, 2, 1])
            phi_x = tf.reshape(phi, [batchsize, out_channels, -1])

            f = tf.matmul(theta_x, phi_x)
            f_softmax = tf.nn.softmax(f, -1)
            y = tf.matmul(f_softmax, g_x)
            y = tf.reshape(y, [batchsize, height, width, out_channels])
            with tf.variable_scope('w') as scope:
                w_y = slim.conv2d(y, in_channels, [1, 1], stride=1, scope='w')
                if is_bn:
                    w_y = slim.batch_norm(w_y)
            z = input_x + w_y
            return z

    def End_Net_weights_init(self):
        """
        Initialize Aggregation Network Weights and Summaries
        """
        W_ext1 = weight_variable([3, 3, 768, 384], "_8")
        W_ext2 = weight_variable([3, 3, 384, 256], "_9")
        end_weights = [W_ext1, W_ext2]
        weight_summaries = []
        for weight_index in range(len(end_weights)):
            with tf.name_scope('weight_%d' % weight_index):
                weight_summaries += variable_summaries(end_weights[weight_index])

        return end_weights, weight_summaries

    def End_Net(self, input_x):
        """
        Define Aggregation Network
        """
        weights, summaries = self.End_Net_weights_init()
        layer8 = conv2d_batchnorm_init(input_x, weights[0], name="conv_8", phase=self.depth_phase, stride=[1, 2, 2, 1])
        nonlocal_block_1 = self.NonLocalBlock(layer8, 384, scope="nonlocal_block_1")
        layer9 = conv2d_batchnorm_init(nonlocal_block_1, weights[1], name="conv_9", phase=self.depth_phase, stride=[1, 2, 2, 1])
        nonlocal_block_2 = self.NonLocalBlock(layer9, 256, scope="nonlocal_block_2")
        return nonlocal_block_2, summaries

    def End_Net_Output(self):
        """
        Computation Graph
        """
        RGB_Net_obj = model.Resnet(self.cam, self.rgb_pahse)
        Depth_Net_obj = model_depth.Depthnet(self.velo, self.depth_phase)
        with tf.variable_scope('ResNet_RGB'):
            with tf.device('/device:GPU:0'):
                output_rgb = RGB_Net_obj.Net()
        with tf.variable_scope('ResNet_Depth'):
            with tf.device('/device:GPU:0'):
                output_depth = Depth_Net_obj.Net()

        with tf.variable_scope('End_Net'):
            layer_combine = tf.concat([output_depth, output_rgb], 3)
            combine_output, summaries = self.End_Net(layer_combine)

        return combine_output, summaries

    def build(self):
        output, summaries = self.End_Net_Output()
        with tf.name_scope('predict_transform_vector'):
            W_tr = weight_variable([1, 2, 256, 128], "W_tr")
            conv_tr = conv2d_batchnorm_init(output, W_tr, name="conv_tr", phase=self.depth_phase,
                                             stride=[1, 1, 1, 1])
            nonlocal_block3 = self.NonLocalBlock(conv_tr, 128, scope='nonlocal_block3')
            nonlocal_block3 = tf.contrib.layers.flatten(nonlocal_block3)

            W_tr_fc = weight_variable_fc([2 * 5 * 128, 3], '_11')
            output_tr = tf.nn.dropout(nonlocal_block3, self.fc_keep_prob)
            vec_tr = tf.matmul(output_tr, W_tr_fc)
            summaries.append(variable_summaries(W_tr))
            summaries.append(variable_summaries(W_tr_fc))

        with tf.name_scope('predict_rotation_vector'):
            W_ro = weight_variable([1, 2, 256, 128], "W_ro")
            conv_ro = conv2d_batchnorm_init(output, W_ro, name="conv_ro", phase=self.depth_phase,
                                             stride=[1, 1, 1, 1])
            nonlocal_block4 = self.NonLocalBlock(conv_ro, 128, scope='nonlocal_block4')
            nonlocal_block4 = tf.contrib.layers.flatten(nonlocal_block4)

            W_ro_fc = weight_variable_fc([2 * 5 * 128, 3], '_12')
            output_ro = tf.nn.dropout(nonlocal_block4, self.fc_keep_prob)
            vec_ro = tf.matmul(output_ro, W_ro_fc)
            summaries.append(variable_summaries(W_ro))
            summaries.append(variable_summaries(W_ro_fc))

        result = tf.concat([vec_tr, vec_ro], 1)

        return result, summaries




