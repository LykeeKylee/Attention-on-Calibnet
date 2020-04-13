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

class RCNN:
    def __init__(self, rgb_inputs, pooled_inputs, lstm_num, lstm_hidden_size,
                 time_step, phase_rgb, phase, fc_keep_prob, keep_prob):
        '''The input's shape is [batch_size, lstm_time_step, input_dims]'''
        self.rgb_inputs = rgb_inputs
        self.pooled_inputs = pooled_inputs
        self.lstm_num = lstm_num
        self.lstm_hidden_size = lstm_hidden_size
        self.time_step = time_step
        self.keep_prob = keep_prob
        self.phase_rgb = phase_rgb
        self.phase = phase
        self.fc_keep_prob = fc_keep_prob

    def get_rnn_inputs(self):
        cnn_output, summaries = self.End_Net_Out(self.rgb_inputs, self.phase_rgb,
                                                 self.pooled_inputs, self.phase)
        cnn_output = tf.contrib.layers.flatten(cnn_output)
        return cnn_output, [summaries]

    def build(self):
        # ResNet
        self.rnn_inputs,summaries = self.get_rnn_inputs()

        # forward layers
        self.rnn_layers_fw = []
        # backward layers
        self.rnn_layers_bw = []
        for layer in range(self.lstm_num):
            self.rnn_layers_fw.append(tf.nn.rnn_cell.GRUCell(num_units=self.lstm_hidden_size, activation=tf.nn.tanh))
            self.rnn_layers_bw.append(tf.nn.rnn_cell.GRUCell(num_units=self.lstm_hidden_size, activation=tf.nn.tanh))

        if self.keep_prob is not None:
            for layer in range(self.lstm_num):
                self.rnn_layers_fw[layer] = tf.nn.rnn_cell.DropoutWrapper(self.rnn_layers_fw[layer],
                                                                          output_keep_prob=self.keep_prob[layer])
                self.rnn_layers_bw[layer] = tf.nn.rnn_cell.DropoutWrapper(self.rnn_layers_bw[layer],
                                                                          output_keep_prob=self.keep_prob[layer])
        self.rnn_layers_fw = tf.nn.rnn_cell.MultiRNNCell(self.rnn_layers_fw)
        self.rnn_layers_bw = tf.nn.rnn_cell.MultiRNNCell(self.rnn_layers_bw)

        time_steps = tf.ones([batch_size, ], dtype=tf.int32) * self.time_step

        self.rnn_inputs = tf.reshape(self.rnn_inputs, shape=(batch_size, self.time_step, int(self.rnn_inputs.shape[-1])))

        with tf.name_scope('lstm_output'):
            lstm_output, lstm_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.rnn_layers_fw,
                                                                      cell_bw=self.rnn_layers_bw,
                                                                      inputs=self.rnn_inputs,
                                                                      sequence_length=time_steps,
                                                                      time_major=False,
                                                                      dtype=tf.float32)
            summaries.append(variable_summaries(lstm_state[0]))
            summaries.append(variable_summaries(lstm_state[1]))

        lstm_output = tf.concat(lstm_output, axis=-1)
        lstm_output = tf.reshape(lstm_output, shape=(batch_size * self.time_step, self.lstm_hidden_size * 2))

        with tf.name_scope('weight_11'):
            W_tr_fc = weight_variable_fc([self.lstm_hidden_size * 2, 3], '_11')
            W_tr_fc = tf.nn.dropout(W_tr_fc, self.fc_keep_prob)
            summaries.append(variable_summaries(W_tr_fc))
            vec_tr = tf.matmul(lstm_output, W_tr_fc)

        with tf.name_scope('weight_12'):
            W_ro_fc = weight_variable_fc([self.lstm_hidden_size * 2, 3], '_12')
            W_ro_fc = tf.nn.dropout(W_ro_fc, self.fc_keep_prob)
            summaries.append(variable_summaries(W_ro_fc))
            vec_ro = tf.matmul(lstm_output, W_ro_fc)

        result = tf.concat([vec_tr, vec_ro], 1)

        return result, summaries

    def End_Net_weights_init(self):

        """
        Initialize Aggregation Network Weights and Summaries
        """
        W_ext1 = weight_variable([3, 3, 768, 384], "_8")
        W_ext2 = weight_variable([3, 3, 384, 384], "_9")
        W_ext3 = weight_variable([1, 2, 384, 384], "_10")

        end_weights = [W_ext1, W_ext2, W_ext3]

        weight_summaries = []

        for weight_index in range(len(end_weights)):
            with tf.name_scope('weight_%d' % weight_index):
                weight_summaries += variable_summaries(end_weights[weight_index])

        return end_weights, weight_summaries

    def End_Net(self,input_x, phase_depth):

        """
        Define Aggregation Network
        """

        weights, summaries = self.End_Net_weights_init()

        layer8 = conv2d_batchnorm_init(input_x, weights[0], name="conv_9", phase=phase_depth, stride=[1, 2, 2, 1])
        layer9 = conv2d_batchnorm_init(layer8, weights[1], name="conv_10", phase=phase_depth, stride=[1, 2, 2, 1])
        layer10 = conv2d_batchnorm_init(layer9, weights[2], name="conv_11", phase=phase_depth, stride=[1, 1, 1, 1])

        # todo layer10
        return layer10, summaries

    def End_Net_Out(self, X1, phase_rgb, pooled_input2, phase):

        """
        Computation Graph
        """
        with tf.variable_scope('ResNet'):
            RGB_Net_obj = model.Resnet(X1, phase_rgb)
            with tf.device('/device:GPU:0'):
                output_rgb = RGB_Net_obj.Net()

        with tf.variable_scope('ResNet_Depth'):
            Depth_Net_obj = model_depth.Depthnet(pooled_input2, phase)
            with tf.device('/device:GPU:0'):
                output_depth = Depth_Net_obj.Net()

        layer_next = tf.concat([output_depth, output_rgb], 3)

        with tf.variable_scope('End_Net'):
            cnn_output, summaries = self.End_Net(layer_next, phase)

        return cnn_output, summaries





