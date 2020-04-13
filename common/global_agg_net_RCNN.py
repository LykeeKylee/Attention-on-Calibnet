import numpy as np
import tensorflow as tf
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

# def End_Net(input_x, phase_depth, keep_prob):
#
#     """
#     Define Aggregation Network
#     """
#
#     weights, summaries = End_Net_weights_init()
#
#     layer8 = conv2d_batchnorm_init(input_x, weights[0], name="conv_9", phase= phase_depth, stride=[1,2,2,1])
#     layer9 = conv2d_batchnorm_init(layer8, weights[1], name="conv_10", phase= phase_depth, stride=[1,2,2,1])
#     layer10 = conv2d_batchnorm_init(layer9, weights[2], name="conv_11", phase= phase_depth, stride=[1,1,1,1])
#
#     layer11_rot = conv2d_batchnorm_init(layer10, weights[3], name="conv_12", phase= phase_depth, stride=[1,1,1,1])
#     layer11_m_rot = tf.reshape(layer11_rot, [batch_size, 3840])
#     layer11_drop_rot = tf.nn.dropout(layer11_m_rot, keep_prob)
#     layer11_vec_rot = (tf.matmul(layer11_drop_rot, weights[4]))
#
#     layer11_tr = conv2d_batchnorm_init(layer10, weights[5], name="conv_13", phase= phase_depth, stride=[1,1,1,1])
#     layer11_m_tr = tf.reshape(layer11_tr, [batch_size, 3840])
#     layer11_drop_tr = tf.nn.dropout(layer11_m_tr, keep_prob)
#     layer11_vec_tr = (tf.matmul(layer11_drop_tr, weights[6]))
#
#     output_vectors = tf.concat([layer11_vec_tr, layer11_vec_rot], 1)
#     return output_vectors, summaries
#
#
# def End_Net_Out(X1, phase_rgb, pooled_input2, phase, keep_prob):
#
#     """
#     Computation Graph
#     """
#
#     RGB_Net_obj = model.Resnet(X1, phase_rgb)
#     Depth_Net_obj = model_depth.Depthnet(pooled_input2, phase)
#
#     with tf.variable_scope('ResNet'):
#         with tf.device('/device:GPU:0'):
#             output_rgb = RGB_Net_obj.Net()
#             output_depth = Depth_Net_obj.Net()
#
#     layer_next = tf.concat([output_rgb, output_depth], 3)
#
#     end_net_op = End_Net(layer_next, phase, keep_prob)
#
#     return end_net_op

# LSTM层
class RCNN:
    def __init__(self, rgb_inputs, pooled_inputs, lstm_num, lstm_hidden_size,
                 time_step, phase_rgb, phase, cnn_keep_prob, keep_prob):
        '''The input's shape is [batch_size, lstm_time_step, input_dims]'''
        self.rgb_inputs = rgb_inputs
        self.pooled_inputs = pooled_inputs
        self.lstm_num = lstm_num
        self.lstm_hiddn_size = lstm_hidden_size
        self.time_step = time_step
        self.keep_prob = keep_prob
        self.phase_rgb = phase_rgb
        self.phase = phase
        self.cnn_keep_prob = cnn_keep_prob



    def get_rnn_inputs(self):
        inputs_rgb, inputs_depth = tf.unstack(self.rgb_inputs, axis=1), tf.unstack(self.pooled_inputs, axis=1)

        rnn_inputs = []
        layer_next = []
        vsummaries = []
        for idx in range(self.time_step):
            layer_next.append(self.Res_Net_Out(inputs_rgb[idx], self.phase_rgb, inputs_depth[idx], self.phase, self.cnn_keep_prob, idx))

        end_net_inputs = []
        for idx in range(self.time_step - 1):
            end_net_inputs.append(tf.concat([layer_next[idx], layer_next[idx + 1]], axis=-1))

            if idx == 0:
                with tf.variable_scope('End_Net'):
                    cnn_output, summaries = self.End_Net(end_net_inputs[idx], self.phase, self.keep_prob)
            else:
                with tf.variable_scope('End_Net', reuse=True):
                    cnn_output, summaries = self.End_Net(end_net_inputs[idx], self.phase, self.keep_prob)
            cnn_output = tf.contrib.layers.flatten(cnn_output, [-1,])
            rnn_inputs.append(cnn_output)
            vsummaries.append(summaries)

            if idx == self.time_step - 2:
                end_net_inputs.append(tf.concat([layer_next[idx + 1], layer_next[idx + 1]], axis=-1))
                with tf.variable_scope('End_Net', reuse=True):
                    cnn_output, summaries = self.End_Net(end_net_inputs[idx + 1], self.phase, self.keep_prob)
                    cnn_output = tf.contrib.layers.flatten(cnn_output, [-1, ])
                    rnn_inputs.append(cnn_output)
                    vsummaries.append(summaries)

        return rnn_inputs, vsummaries


    def build(self):
        # ResNet
        self.rgb_inputs = tf.reshape(self.rgb_inputs, (batch_size, self.time_step, IMG_HT, IMG_WDT, 3))
        self.pooled_inputs = tf.reshape(self.pooled_inputs, (batch_size, self.time_step, IMG_HT, IMG_WDT, 1))
        self.rnn_inputs, summaries = self.get_rnn_inputs()

        # lstm layers
        self.rnn_layers = [tf.nn.rnn_cell.LSTMCell(self.lstm_hiddn_size) for _ in range(self.lstm_num)]

        # whether to use keep_prob
        if self.keep_prob is not None:
            for i in range(self.lstm_num):
                self.rnn_layers[i] = tf.nn.rnn_cell.DropoutWrapper(self.rnn_layers[i], output_keep_prob=self.keep_prob[i])

        time_steps = tf.ones([batch_size, ], dtype=tf.int32) * self.time_step

        with tf.name_scope('lstm_output'):
            self.multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(self.rnn_layers)
            lstm_output, lstm_state = tf.nn.static_rnn(self.multi_rnn_cell, self.rnn_inputs,
                                                               sequence_length=time_steps, dtype=tf.float32, scope='lstm_output')
            summaries.append(variable_summaries(lstm_state))

        lstm_output = tf.concat(lstm_output, axis=1)
        lstm_output = tf.reshape(lstm_output, shape=(batch_size * self.time_step, self.lstm_hiddn_size))

        with tf.name_scope('weight_11'):
            W_tr_fc = weight_variable_fc([self.lstm_hiddn_size, 3], '_11')
            summaries.append(variable_summaries(W_tr_fc))
        drop_tr = tf.nn.dropout(lstm_output, self.cnn_keep_prob)
        vec_tr = tf.matmul(drop_tr, W_tr_fc)

        with tf.name_scope('weight_12'):
            W_ro_fc = weight_variable_fc([self.lstm_hiddn_size, 3], '_12')
            summaries.append(variable_summaries(W_ro_fc))
        drop_rot = tf.nn.dropout(lstm_output, self.cnn_keep_prob)
        vec_ro = tf.matmul(drop_rot, W_ro_fc)

        result = tf.concat([vec_tr, vec_ro], 1)

        return result, summaries

    def End_Net_weights_init(self):

        """
        Initialize Aggregation Network Weights and Summaries
        """

        # W_ext1 = weight_variable([3, 3, 768, 384], "_8")
        # W_ext2 = weight_variable([3, 3, 384, 384], "_9")
        # W_ext3 = weight_variable([1, 2, 384, 384], "_10")
        # todo new
        W_ext1 = weight_variable([3, 3, 1536, 768], "_8")
        W_ext2 = weight_variable([3, 3, 768, 384], "_9")
        W_ext3 = weight_variable([1, 2, 384, 384], "_10")

        # W_ext4_rot = weight_variable([1, 1, 384, 384], "_11")
        # W_fc_rot = weight_variable_fc([3840, 3], "_12")
        #
        # W_ext4_tr = weight_variable([1, 1, 384, 384], "_13")
        # W_fc_tr = weight_variable_fc([3840, 3], "_14")

        # end_weights = [W_ext1, W_ext2, W_ext3, W_ext4_rot, W_fc_rot, W_ext4_tr, W_fc_tr]
        end_weights = [W_ext1, W_ext2, W_ext3]

        weight_summaries = []

        for weight_index in range(len(end_weights)):
            with tf.name_scope('weight_%d' % weight_index):
                weight_summaries += variable_summaries(end_weights[weight_index])

        return end_weights, weight_summaries

    def End_Net(self,input_x, phase_depth, keep_prob):

        """
        Define Aggregation Network
        """

        weights, summaries = self.End_Net_weights_init()

        layer8 = conv2d_batchnorm_init(input_x, weights[0], name="conv_9", phase=phase_depth, stride=[1, 2, 2, 1])
        layer9 = conv2d_batchnorm_init(layer8, weights[1], name="conv_10", phase=phase_depth, stride=[1, 2, 2, 1])
        layer10 = conv2d_batchnorm_init(layer9, weights[2], name="conv_11", phase=phase_depth, stride=[1, 1, 1, 1])

        return layer10, summaries

    def Res_Net_Out(self, X1, phase_rgb, pooled_input2, phase, keep_prob, idx):

        """
        Computation Graph
        """
        if idx == 0:
            with tf.variable_scope('ResNet'):

                RGB_Net_obj = model.Resnet(X1, phase_rgb)
                Depth_Net_obj = model_depth.Depthnet(pooled_input2, phase)

                with tf.device('/device:GPU:0'):
                    output_rgb = RGB_Net_obj.Net()
                    output_depth = Depth_Net_obj.Net()


            layer_next = tf.concat([output_rgb, output_depth], 3)

            # with tf.variable_scope('End_Net'):
            #     cnn_output = self.End_Net(layer_next, phase, keep_prob)

        else:
            with tf.variable_scope('ResNet', reuse=True):
                RGB_Net_obj = model.Resnet(X1, phase_rgb)
                Depth_Net_obj = model_depth.Depthnet(pooled_input2, phase)
                with tf.device('/device:GPU:0'):
                    output_rgb = RGB_Net_obj.Net()
                    output_depth = Depth_Net_obj.Net()

            layer_next = tf.concat([output_rgb, output_depth], 3)

            # with tf.variable_scope('End_Net', reuse=True):
            #     cnn_output = self.End_Net(layer_next, phase, keep_prob)

        return layer_next





