import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config_res as config
from common.cnn_utils_res import *

import common.Net as model
import tqdm

batch_size = config.net_params['batch_size'] * config.net_params['time_step']
current_epoch = config.net_params['load_epoch']
IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']

class AMCF:
    def __init__(self, rgb_inputs, pooled_inputs, phase_rgb, phase_depth, fc_keep_prob):
        '''The input's shape is [batch_size, lstm_time_step, input_dims]'''
        self.rgb_inputs = rgb_inputs
        self.pooled_inputs = pooled_inputs
        self.phase_rgb = phase_rgb
        self.phase = phase_depth
        self.fc_keep_prob = fc_keep_prob

    def build(self):
        self.rgb_inputs = tf.reshape(self.rgb_inputs, (batch_size, IMG_HT, IMG_WDT, 3))

        output, summaries = self.End_Net_Out(self.rgb_inputs, self.pooled_inputs, self.phase_rgb, self.phase)
        output = tf.reshape(output, [batch_size, 2 * 5 * 384])

        with tf.name_scope('weight_11'):
            W_tr_fc = weight_variable_fc([2 * 5 * 384, 3], '_11')
            output_tr = tf.nn.dropout(output, self.fc_keep_prob)
            summaries.append(variable_summaries(W_tr_fc))
            vec_tr = tf.matmul(output_tr, W_tr_fc)

        with tf.name_scope('weight_12'):
            W_ro_fc = weight_variable_fc([2 * 5 * 384, 3], '_12')
            output_ro = tf.nn.dropout(output, self.fc_keep_prob)
            summaries.append(variable_summaries(W_ro_fc))
            vec_ro = tf.matmul(output_ro, W_ro_fc)

        result = tf.concat([vec_tr, vec_ro], 1)
        # with tf.name_scope('weight_fc'):
        #     W_fc = weight_variable_fc([2 * 5 * 384, 6], '_11')
        #     output = tf.nn.dropout(output, self.fc_keep_prob)
        #     summaries.append(variable_summaries(W_fc))
        #     result = tf.matmul(output, W_fc)

        return result, summaries

    def leaky_relu(self, x, leak=0.2):
        return (0.5 * (1 + leak)) * x + (0.5 * (1 - leak)) * tf.abs(x)

    def End_Net(self, rgb, depth, phase_depth, phase_rgb):

        """
        Define Aggregation Network
        """
        summaries = []
        combine = tf.concat([depth, rgb], -1)
        # combine = tf.reshape(combine, [batch_size, Frame, 6, 20, 768])
        with tf.variable_scope("down"):
            Wd0 = weight_variable([1, 1, 768, 384], "d0")
            Wd1 = weight_variable([3, 3, 768, 384], "d1")
            Wd2 = weight_variable([3, 3, 384, 384], "d2")

            summaries += [variable_summaries(Wd0), variable_summaries(Wd1), variable_summaries(Wd2)]
            
            down1 = tf.nn.conv2d(combine, Wd1, [1, 2, 2, 1], "SAME")
            bn1 = tf.contrib.layers.batch_norm(down1, is_training=phase_depth)
            relu1 = tf.nn.relu(bn1)
            
            down2 = tf.nn.conv2d(relu1, Wd2, [1, 2, 2, 1], "SAME")
            bn2 = tf.contrib.layers.batch_norm(down2, is_training=phase_depth)
            relu2 = tf.nn.relu(bn2)
            
            down0 = tf.nn.conv2d(combine, Wd0, [1, 4, 4, 1], "SAME")
            
            out_1 = tf.nn.relu(down0 + relu2)
            
            
        with tf.variable_scope("GC"):
            for i in range(1):
                if i == 0:
                    input1 = out_1
                
                with tf.variable_scope("non_local" + str(i)):
                    W_0 = weight_variable([1, 1, 384, 1], "_0")
                    summaries.append(variable_summaries(W_0))
        
                    map = tf.nn.conv2d(input1, W_0, [1, 1, 1, 1], "SAME")
                    map = tf.nn.softmax(map)
                    map = tf.reshape(map, [batch_size, 2 * 5, 1])
                    map_t = tf.transpose(map, [0, 2, 1])
        
                    input1_r = tf.reshape(input1, [batch_size, 2 * 5, 384])
                    attention_pooling = tf.matmul(map_t, input1_r)
        
                with tf.variable_scope("sqeeze" + str(i)):
                    W_1 = weight_variable_fc([batch_size, 384, 96], "_1")
                    sqeeze = tf.matmul(attention_pooling, W_1)
                    bn = tf.contrib.layers.batch_norm(sqeeze, is_training=phase_depth)
                    relu_1 = tf.nn.relu(bn)
        
                    W_2 = weight_variable_fc([batch_size, 96, 384], "_2")
                    transform = tf.matmul(relu_1, W_2)
                    transform = tf.reshape(transform, [batch_size, 1, 1, 384])

                summaries += [variable_summaries(W_0), variable_summaries(W_1), variable_summaries(W_2)]
                out = input1 + transform
                input1 = out

        # todo layer10
        return out, summaries

    def End_Net_Out(self, X1, pooled_input2, phase_rgb, phase):

        """
        Computation Graph
        """
        with tf.variable_scope('ResNet34_RGB'):
            RGB_Net_obj = model.ResNet34_RGB(X1, phase_rgb)
            with tf.device('/device:GPU:0'):
                output_rgb, summary1 = RGB_Net_obj.Net()
                # output_rgb = tf.reshape(output_rgb, (batch_size * Frame, 6, 20, 512))

        with tf.variable_scope('ResNet34_Depth'):
            Depth_Net_obj = model.ResNet34_Depth(pooled_input2, phase)
            with tf.device('/device:GPU:0'):
                output_depth, summary2 = Depth_Net_obj.Net()
                # output_depth = tf.reshape(output_depth, [batch_size, Frame, 6, 20, 256])

        with tf.variable_scope('End_Net'):
            with tf.device('/device:GPU:0'):
                cnn_output, summaries = self.End_Net(output_rgb, output_depth, phase, phase_rgb)

        return cnn_output, summaries + summary1 + summary2





