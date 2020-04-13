import numpy as np
import tensorflow as tf
import json
from common.cnn_utils_res import *

import config_res

with open(config_res.paths['resnet_params_path']) as f_in:
    parameters = json.load(f_in)

class Depthnet:

    def __init__(self, input_y, phase, parameters = parameters):

        self.input_y = input_y
        self.phase = phase
        self.parameters = parameters
        self.layer_zero
        self.layer
        self.Net

    def Net(self):
        layer_zero_out = self.layer_zero(self.input_y)

        current_output = layer_zero_out

        for layer_idx in range(1,5):
            layer_out = self.layer(current_output, layer_idx)
            current_output = layer_out

        return current_output

    def layer_zero(self, layer_input):

        layer_dict =self.parameters['layer0']
        bl_str = "block_1"

        W = np.array(layer_dict[bl_str]['conv1']['weight'], dtype = np.float32)
        bn_mov_mean = np.array(layer_dict[bl_str]['bn1']['running_mean'], dtype = np.float32)
        bn_mov_var = np.array(layer_dict[bl_str]['bn1']['running_var'], dtype = np.float32)
        bn_gamma = np.array(layer_dict[bl_str]['bn1']['weight'], dtype = np.float32)
        bn_beta = np.array(layer_dict[bl_str]['bn1']['bias'], dtype = np.float32)

        shapex = W.shape
        W_conv = weight_variable([shapex[0], shapex[1], 1, shapex[3]/2], "_depth_0")
        # out = conv2d_batchnorm(layer_input, W_conv, "layer_depth_0", self.phase, bn_beta, bn_gamma, bn_mov_mean, bn_mov_var, [1,2,2,1], True)
        out = conv2d_batchnorm_init(layer_input, W_conv, "layer_depth_0", self.phase, [1,2,2,1], True)

        out = tf.nn.max_pool(out, [1,3,3,1], strides=[1,2,2,1], padding="SAME")

        # print('layer0', out.shape)
        return out

    def layer(self, layer_input, layer_no):
        layer_dict = self.parameters['layer%d'%layer_no]

        cur = layer_input
        res = layer_input

        for b_no in range(1,3):
            bl_str = "block_%d"%b_no

            stride = [0,0]
            if(b_no == 1):
                stride = [2,1]
            else:
                stride = [1,1]

            # for in_bno in range(1,3):

            W1 = np.array(layer_dict[bl_str]['conv1']['weight'], dtype = np.float32)
            bn_mov_mean1 = np.array(layer_dict[bl_str]['bn1']['running_mean'], dtype = np.float32)
            bn_mov_var1 = np.array(layer_dict[bl_str]['bn1']['running_var'], dtype = np.float32)
            bn_gamma1 = np.array(layer_dict[bl_str]['bn1']['weight'], dtype = np.float32)
            bn_beta1 = np.array(layer_dict[bl_str]['bn1']['bias'], dtype = np.float32)

            W2 = np.array(layer_dict[bl_str]['conv2']['weight'], dtype = np.float32)
            bn_mov_mean2 = np.array(layer_dict[bl_str]['bn2']['running_mean'], dtype = np.float32)
            bn_mov_var2 = np.array(layer_dict[bl_str]['bn2']['running_var'], dtype = np.float32)
            bn_gamma2 = np.array(layer_dict[bl_str]['bn2']['weight'], dtype = np.float32)
            bn_beta2 = np.array(layer_dict[bl_str]['bn2']['bias'], dtype = np.float32)

            # W_conv1 = init_weights(W1, "_l_%d_bl_%d_no_%d"%(layer_no,b_no, 1), False)
            # W_conv2 = init_weights(W2, "_l_%d_bl_%d_no_%d"%(layer_no,b_no, 2), False)

            shapex1 = W1.shape
            shapex2 = W2.shape

            W_conv1 = weight_variable([shapex1[0], shapex1[1], shapex1[2]/2, shapex1[3]/2], "dep_l_%d_bl_%d_no_%d"%(layer_no,b_no, 1))
            W_conv2 = weight_variable([shapex2[0], shapex2[1], shapex2[2]/2, shapex2[3]/2], "dep_l_%d_bl_%d_no_%d"%(layer_no,b_no, 2))

            # out1 = conv2d_batchnorm(cur, W_conv1, "layer_%d_%d_1"%(layer_no,b_no), self.phase, bn_beta1, bn_gamma1, bn_mov_mean1, bn_mov_var1, [1,stride[0],stride[0],1], False)

            out1 = conv2d_batchnorm_init(cur, W_conv1, "dep_layer_%d_%d_1"%(layer_no,b_no), self.phase, [1,stride[0],stride[0],1], False)

            # print("layer_%d_%d_1"%(layer_no,b_no), out1.shape)

            """ if layer1 no downsample, so stride 2,1 then 1,1 """
            """else stride 2,1 then downsample then 1,1 """

            if(layer_no > 1 and b_no == 1):
                downsample_dict = self.parameters['layer%d_downsample'%layer_no]
                W_dn = np.array(downsample_dict['block_1']['conv']['weight'], dtype = np.float32)
                bn_mov_mean_dn = np.array(downsample_dict['block_1']['bn']['running_mean'], dtype = np.float32)
                bn_mov_var_dn = np.array(downsample_dict['block_1']['bn']['running_var'], dtype = np.float32)
                bn_gamma_dn = np.array(downsample_dict['block_1']['bn']['weight'], dtype = np.float32)
                bn_beta_dn = np.array(downsample_dict['block_1']['bn']['bias'], dtype = np.float32)

                # W_conv_dn = init_weights(W_dn, "downsample_%d"%(layer_no), False)
                # res = conv2d_batchnorm(res, W_conv_dn, "layer_dn_%d"%(layer_no), self.phase, bn_beta_dn, bn_gamma_dn, bn_mov_mean_dn, bn_mov_var_dn, [1,2,2,1], False)

                shapex = W_dn.shape

                W_conv_dn = weight_variable([shapex[0], shapex[1], shapex[2]/2, shapex[3]/2], "dep_downsample_%d"%(layer_no))
                res = conv2d_batchnorm_init(res, W_conv_dn, "dep_layer_dn_%d"%(layer_no), self.phase, [1,2,2,1], False)

                # print("downsample_layer_%d_%d_1"%(layer_no,b_no), res.shape)

                out1 = tf.nn.relu(out1 + res)

            else:
                out1 = tf.nn.relu(out1)

            out2 = conv2d_batchnorm_init(out1, W_conv2, "dep_layer_%d_%d_2"%(layer_no,b_no), self.phase, [1,stride[1],stride[1],1], True)
            # print("layer_%d_%d_2"%(layer_no,b_no), out2.shape)
            cur = out2

        return cur

# import numpy as np
# import tensorflow as tf
# import json
# from common.cnn_utils_res import *
#
# import config_res as config
#
# IMG_HT = config.depth_img_params['IMG_HT']
# IMG_WDT = config.depth_img_params['IMG_WDT']
#
# class Depthnet:
#
#     def __init__(self, input):
#         self.input = input
#
#     def Net(self):
#         with tf.variable_scope("layer_0"):
#             w = tf.get_variable("weight", [7, 7, 4, 96], initializer=tf.truncated_normal_initializer(stddev=0.1))
#             layer_0 = tf.nn.conv2d(self.input, filter=w, strides=[1, 2, 2, 1], padding="SAME")
#             layer_0 = tf.layers.batch_normalization(layer_0, training=True)
#             layer_0 = tf.nn.relu(layer_0)
#             layer_0 = tf.nn.max_pool(layer_0,
#                                      ksize=[1, 3, 3, 1],
#                                      strides=[1, 2, 2, 1],
#                                      padding="SAME")
#
#         layer_1 = self.Block(layer_0, 96, 96, "1")
#         layer_2 = self.Block(layer_1, 96, 192, "2")
#         layer_3 = self.Block(layer_2, 192, 384, "3")
#         layer_4 = self.Block(layer_3, 384, 768, "4")
#
#         return layer_4
#
#     def Block(self, layer, input_channels, output_channels, no):
#         with tf.variable_scope("layer_" + no):
#             layer0 = layer
#             layer1 = layer
#
#             w_1 = tf.get_variable("weight_1", [3, 3, input_channels, output_channels], initializer=tf.truncated_normal_initializer(stddev=0.1))
#             layer1 = tf.nn.conv2d(layer1, w_1, [1, 2, 2, 1], "SAME")
#             layer1 = tf.layers.batch_normalization(layer1, training=True)
#             layer1 = tf.nn.relu(layer1)
#
#             w_2 = tf.get_variable("weight_2", [3, 3, output_channels, output_channels], initializer=tf.truncated_normal_initializer(stddev=0.1))
#             layer1 = tf.nn.conv2d(layer1, w_2, [1, 1, 1, 1], "SAME")
#             layer1 = tf.layers.batch_normalization(layer1, training=True)
#
#             w_0 = tf.get_variable("weight_0", [1, 1, input_channels, output_channels], initializer=tf.truncated_normal_initializer(stddev=0.1))
#             layer0 = tf.nn.conv2d(layer0, w_0, [1, 2, 2, 1], "SAME")
#             layer0 = tf.layers.batch_normalization(layer0, training=True)
#
#             out = tf.nn.relu(layer0 + layer1)
#             return out
