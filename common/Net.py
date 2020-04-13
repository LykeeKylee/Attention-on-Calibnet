import tensorflow as tf
from common.cnn_utils_res import *
from math import *

import config_res as config

IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
# Frame = config.net_params['Frame']
batch_size = config.net_params['batch_size'] * config.net_params['time_step']


# class ResNet50_RGB:
#
#     def __init__(self, input, to_train):
#         self.input = input
#         self.to_train = False
#
#     def Net(self):
#         with tf.variable_scope("conv0"):
#             W_1 = weight_variable([1, 7, 7, 3, 32], "_1", to_train=True)
#             conv3d_1 = tf.nn.conv3d(self.input, W_1, [1, 1, 2, 2, 1], "SAME")
#
#             bn_1 = tf.contrib.layers.batch_norm(conv3d_1, is_training=self.to_train)
#             relu_1 = tf.nn.relu(bn_1)
#             max_pool_1 = tf.nn.max_pool3d(relu_1, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], "SAME")
#
#         conv1 = self.Res_Block(max_pool_1, 1, 32, 128, 3, "1")
#         conv2 = self.Res_Block(conv1, 1, 64, 256, 4, "2")
#         conv3 = self.Res_Block(conv2, 1, 128, 512, 6, "3")
#         conv4 = self.Res_Block(conv3, 1, 256, 1024, 3, "4")
#
#         avg_pool_1 = tf.nn.avg_pool3d(conv4, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], "SAME")
#
#         return avg_pool_1
#
#     def Res_Block(self, layer_input, depth, input_channels, output_channels, block_num, no, to_train=True):
#         input_ = layer_input
#         with tf.variable_scope("conv%s" % no):
#             for i in range(block_num):
#                 with tf.variable_scope("conv%s_%d" % (no, i)):
#                     if no != "1":
#                         if i != 0:
#                             W_1 = weight_variable([depth, 1, 1, output_channels, input_channels], "_1", to_train=True)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
#                         else:
#                             W_0 = weight_variable([depth, 1, 1, input_channels * 2, output_channels], "_0", to_train=True)
#                             conv3d_0 = tf.nn.conv3d(input_, W_0, [1, 1, 2, 2, 1], "SAME")
#                             bn_0 = tf.contrib.layers.batch_norm(conv3d_0, is_training=self.to_train)
#
#                             W_1 = weight_variable([depth, 1, 1, input_channels * 2, input_channels], "_1", to_train=True)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 2, 2, 1], "SAME")
#
#                     else:
#                         if i != 0:
#                             W_1 = weight_variable([depth, 1, 1, output_channels, input_channels], "_1", to_train=True)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
#                         else:
#                             W_0 = weight_variable([depth, 1, 1, input_channels, output_channels], "_0", to_train=True)
#                             conv3d_0 = tf.nn.conv3d(input_, W_0, [1, 1, 1, 1, 1], "SAME")
#                             bn_0 = tf.contrib.layers.batch_norm(conv3d_0, is_training=self.to_train)
#
#                             W_1 = weight_variable([depth, 1, 1, input_channels, input_channels], "_1", to_train=True)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
#
#                     W_2 = weight_variable([depth, 3, 3, input_channels, input_channels], "_2", to_train=True)
#                     W_3 = weight_variable([depth, 1, 1, input_channels, output_channels], '_3', to_train=True)
#
#                     bn_1 = tf.contrib.layers.batch_norm(conv3d_1, is_training=self.to_train)
#                     relu_1 = tf.nn.relu(bn_1)
#
#                     conv3d_2 = tf.nn.conv3d(relu_1, W_2, [1, 1, 1, 1, 1], "SAME")
#                     bn_2 = tf.contrib.layers.batch_norm(conv3d_2, is_training=self.to_train)
#                     relu_2 = tf.nn.relu(bn_2)
#
#                     conv3d_3 = tf.nn.conv3d(relu_2, W_3, [1, 1, 1, 1, 1], "SAME")
#                     bn_3 = tf.contrib.layers.batch_norm(conv3d_3, is_training=self.to_train)
#                     relu_3 = tf.nn.relu(bn_3)
#
#                     if i != 0:
#                         add = tf.nn.relu(tf.add(relu_3, input_))
#                     else:
#                         add = tf.nn.relu(tf.add(relu_3, bn_0))
#
#                     input_ = add
#
#             return input_
#
# # class ResNet50_Depth:
# #
# #     def __init__(self, input, to_train):
# #         self.input = input
# #         self.to_train = to_train
# #
# #     def Net(self):
# #         with tf.variable_scope("conv0"):
# #             W_1 = weight_variable([5, 5, 1, 32], "_1", to_train=True)
# #             conv2d_1 = tf.nn.conv2d(self.input, W_1, [1, 2, 2, 1], "SAME")
# #             bn_1 = tf.contrib.layers.batch_norm(conv2d_1, is_training=self.to_train)
# #             relu_1 = tf.nn.relu(bn_1)
# #             max_pool_1 = tf.nn.max_pool3d(relu_1, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
# #
# #         conv1 = self.Res_Block(max_pool_1, 1, 32, 128, 3, "1")
# #         conv2 = self.Res_Block(conv1, 1, 64, 256, 4, "2")
# #         conv3 = self.Res_Block(conv2, 1, 128, 512, 6, "3")
# #         conv4 = self.Res_Block(conv3, 1, 256, 1024, 3, "4")
# #
# #         return conv4
# #
# #     def Res_Block(self, layer_input, depth, input_channels, output_channels, block_num, no, to_train=True):
# #         input_ = layer_input
# #         with tf.variable_scope("conv%s" % no):
# #             for i in range(block_num):
# #                 with tf.variable_scope("conv%s_%d" % (no, i)):
# #                     if no != "1":
# #                         if i != 0:
# #                             W_1 = weight_variable([1, 1, output_channels, input_channels], "_1", to_train=True)
# #                             conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 1, 1, 1], "SAME")
# #                         else:
# #                             W_0 = weight_variable([1, 1, input_channels * 2, output_channels], "_0", to_train=True)
# #                             conv2d_0 = tf.nn.conv2d(input_, W_0, [1, 2, 2, 1], "SAME")
# #                             bn_0 = tf.contrib.layers.batch_norm(conv2d_0, is_training=self.to_train)
# #
# #                             W_1 = weight_variable([1, 1, input_channels * 2, input_channels], "_1", to_train=True)
# #                             conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 2, 2, 1], "SAME")
# #
# #                     else:
# #                         if i != 0:
# #                             W_1 = weight_variable([1, 1, output_channels, input_channels], "_1", to_train=True)
# #                             conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 1, 1, 1], "SAME")
# #                         else:
# #                             W_0 = weight_variable([1, 1, input_channels, output_channels], "_0", to_train=True)
# #                             conv2d_0 = tf.nn.conv2d(input_, W_0, [1, 1, 1, 1], "SAME")
# #                             bn_0 = tf.contrib.layers.batch_norm(conv2d_0, is_training=self.to_train)
# #
# #                             W_1 = weight_variable([1, 1, input_channels, input_channels], "_1", to_train=True)
# #                             conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 1, 1, 1], "SAME")
# #
# #                     W_2 = weight_variable([3, 3, input_channels, input_channels], "_2", to_train=True)
# #                     W_3 = weight_variable([1, 1, input_channels, output_channels], '_3', to_train=True)
# #
# #                     bn_1 = tf.contrib.layers.batch_norm(conv2d_1, is_training=self.to_train)
# #                     relu_1 = tf.nn.relu(bn_1)
# #
# #                     conv2d_2 = tf.nn.conv2d(relu_1, W_2, [1, 1, 1, 1], "SAME")
# #                     bn_2 = tf.contrib.layers.batch_norm(conv2d_2, is_training=self.to_train)
# #                     relu_2 = tf.nn.relu(bn_2)
# #
# #                     conv2d_3 = tf.nn.conv2d(relu_2, W_3, [1, 1, 1, 1], "SAME")
# #                     bn_3 = tf.contrib.layers.batch_norm(conv2d_3, is_training=self.to_train)
# #                     relu_3 = tf.nn.relu(bn_3)
# #
# #                     if i != 0:
# #                         add = tf.nn.relu(tf.add(input_, relu_3))
# #                     else:
# #                         add = tf.nn.relu(tf.add(input_, bn_0))
# #
# #                     input_ = add
# #
# #             return input_
#
# class ResNet50_Depth:
#
#     def __init__(self, input, to_train):
#         self.input = input
#         self.to_train = False
#
#     def Net(self):
#         with tf.variable_scope("conv0"):
#             W_1 = weight_variable([1, 5, 5, 1, 16], "_1", to_train=True)
#             conv3d_1 = tf.nn.conv3d(self.input, W_1, [1, 1, 2, 2, 1], "SAME")
#             bn_1 = tf.contrib.layers.batch_norm(conv3d_1, is_training=self.to_train)
#             relu_1 = tf.nn.relu(bn_1)
#             max_pool_1 = tf.nn.max_pool3d(relu_1, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], "SAME")
#
#             conv1 = self.Res_Block(max_pool_1, 1, 16, 64, 3, "1")
#             conv2 = self.Res_Block(conv1, 1, 32, 128, 4, "2")
#             conv3 = self.Res_Block(conv2, 1, 64, 256, 6, "3")
#             conv4 = self.Res_Block(conv3, 1, 128, 512, 3, "4")
#
#         avg_pool_1 = tf.nn.avg_pool3d(conv4, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], "SAME")
#
#         return avg_pool_1
#
#     def Res_Block(self, layer_input, depth, input_channels, output_channels, block_num, no, to_train=True):
#         input_ = layer_input
#         with tf.variable_scope("conv%s" % no):
#             for i in range(block_num):
#                 with tf.variable_scope("conv%s_%d" % (no, i)):
#                     if no != "1":
#                         if i != 0:
#                             W_1 = weight_variable([depth, 1, 1, output_channels, input_channels], "_1", to_train=True)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
#                         else:
#                             W_0 = weight_variable([depth, 1, 1, input_channels * 2, output_channels], "_0", to_train=True)
#                             conv3d_0 = tf.nn.conv3d(input_, W_0, [1, 1, 2, 2, 1], "SAME")
#                             bn_0 = tf.contrib.layers.batch_norm(conv3d_0, is_training=self.to_train)
#
#                             W_1 = weight_variable([depth, 1, 1, input_channels * 2, input_channels], "_1", to_train=True)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 2, 2, 1], "SAME")
#
#                     else:
#                         if i != 0:
#                             W_1 = weight_variable([depth, 1, 1, output_channels, input_channels], "_1", to_train=True)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
#                         else:
#                             W_0 = weight_variable([depth, 1, 1, input_channels, output_channels], "_0", to_train=True)
#                             conv3d_0 = tf.nn.conv3d(input_, W_0, [1, 1, 1, 1, 1], "SAME")
#                             bn_0 = tf.contrib.layers.batch_norm(conv3d_0, is_training=self.to_train)
#
#                             W_1 = weight_variable([depth, 1, 1, input_channels, input_channels], "_1", to_train=True)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
#
#                     W_2 = weight_variable([depth, 3, 3, input_channels, input_channels], "_2", to_train=True)
#                     W_3 = weight_variable([depth, 1, 1, input_channels, output_channels], '_3', to_train=True)
#
#                     bn_1 = tf.contrib.layers.batch_norm(conv3d_1, is_training=self.to_train)
#                     relu_1 = tf.nn.relu(bn_1)
#
#                     conv3d_2 = tf.nn.conv3d(relu_1, W_2, [1, 1, 1, 1, 1], "SAME")
#                     bn_2 = tf.contrib.layers.batch_norm(conv3d_2, is_training=self.to_train)
#                     relu_2 = tf.nn.relu(bn_2)
#
#                     conv3d_3 = tf.nn.conv3d(relu_2, W_3, [1, 1, 1, 1, 1], "SAME")
#                     bn_3 = tf.contrib.layers.batch_norm(conv3d_3, is_training=self.to_train)
#                     relu_3 = tf.nn.relu(bn_3)
#
#                     if i != 0:
#                         add = tf.nn.relu(tf.add(relu_3, input_))
#                     else:
#                         add = tf.nn.relu(tf.add(relu_3, bn_0))
#
#                     input_ = add
#
#             return input_



# class ResNet34_RGB:
# 
#     def __init__(self, input, to_train):
#         self.input = input
#         self.to_train = to_train
#         self.summary = []
# 
#     def Net(self):
#         with tf.variable_scope("conv0"):
#             W_1 = weight_variable([7, 7, 3, 64], "_1")
#             conv3d_1 = tf.nn.conv3d(self.input, W_1, [1, 1, 2, 2, 1], "SAME")
#             self.summary.append(variable_summaries(W_1))
# 
#             bn_1 = tf.contrib.layers.batch_norm(conv3d_1, is_training=self.to_train)
#             relu_1 = tf.nn.relu(bn_1)
# 
#             h, w = int(relu_1.shape[-3]), int(relu_1.shape[-2])
#             relu_1 = tf.reshape(relu_1, (batch_size * Frame, h, w, 64))
#             max_pool_1 = tf.nn.max_pool(relu_1, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
#             h, w = int(max_pool_1.shape[-3]), int(max_pool_1.shape[-2])
#             max_pool_1 = tf.reshape(max_pool_1, [batch_size, Frame, h, w, 64])
# 
#         conv1 = self.Res_Block(max_pool_1, 2, 64, 64, 2, "1")
#         conv2 = self.Res_Block(conv1, 2, 64, 128, 2, "2")
#         conv3 = self.Res_Block(conv2, 2, 128, 256, 2, "3")
#         conv4 = self.Res_Block(conv3, 2, 256, 512, 2, "4")
# 
#         h, w = int(conv4.shape[-3]), int(conv4.shape[-2])
#         conv4 = tf.reshape(conv4, [batch_size * Frame, h, w, 512])
#         avg_pool_1 = tf.nn.avg_pool(conv4, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
#         h, w = int(avg_pool_1.shape[-3]), int(avg_pool_1.shape[-2])
#         avg_pool_1 = tf.reshape(avg_pool_1, [batch_size, Frame, h, w, 512])
# 
#         return avg_pool_1, self.summary
# 
#     def Res_Block(self, layer_input, depth, input_channels, output_channels, block_num, no):
#         input_ = layer_input
#         with tf.variable_scope("conv%s" % no):
#             for i in range(block_num):
#                 with tf.variable_scope("conv%s_%d" % (no, i)):
#                     if i != 0:
#                         W_1 = weight_variable([depth, 3, 3, output_channels, output_channels], "_1")
#                         conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
# 
#                     else:
#                         W_0 = weight_variable([1, 1, input_channels, output_channels], "_0")
#                         W_1 = weight_variable([depth, 3, 3, input_channels, output_channels], "_1")
#                         if no != "1":
#                             h, w = int(input_.shape[-3]), int(input_.shape[-2])
#                             input_2d = tf.reshape(input_, [batch_size * Frame, h, w, input_channels])
#                             conv3d_0 = tf.nn.conv2d(input_2d, W_0, [1, 2, 2, 1], "SAME")
#                             bn_0 = tf.contrib.layers.batch_norm(conv3d_0, is_training=self.to_train)
#                             h, w = int(bn_0.shape[-3]), int(bn_0.shape[-2])
#                             bn_0 = tf.reshape(bn_0, [batch_size, Frame, h, w, output_channels])
# 
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 2, 2, 1], "SAME")
#                         else:
#                             h, w = int(input_.shape[-3]), int(input_.shape[-2])
#                             input_2d = tf.reshape(input_, [batch_size * Frame, h, w, input_channels])
#                             conv3d_0 = tf.nn.conv2d(input_2d, W_0, [1, 1, 1, 1], "SAME")
#                             bn_0 = tf.contrib.layers.batch_norm(conv3d_0, is_training=self.to_train)
#                             h, w = int(bn_0.shape[-3]), int(bn_0.shape[-2])
#                             bn_0 = tf.reshape(bn_0, [batch_size, Frame, h, w, output_channels])
# 
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
# 
#                     W_2 = weight_variable([depth, 3, 3, output_channels, output_channels], "_2")
# 
#                     self.summary += [variable_summaries(W_0), variable_summaries(W_1), variable_summaries(W_2)]
# 
#                     bn_1 = tf.contrib.layers.batch_norm(conv3d_1, is_training=self.to_train)
#                     relu_1 = tf.nn.relu(bn_1)
# 
#                     conv3d_2 = tf.nn.conv3d(relu_1, W_2, [1, 1, 1, 1, 1], "SAME")
#                     bn_2 = tf.contrib.layers.batch_norm(conv3d_2, is_training=self.to_train)
#                     relu_2 = tf.nn.relu(bn_2)
# 
#                     if i != 0:
#                         add = tf.nn.relu(tf.add(relu_2, input_))
#                     else:
#                         add = tf.nn.relu(tf.add(relu_2, bn_0))
# 
#                     input_ = add
#             return input_
        
class ResNet34_RGB:
    def __init__(self, input, to_train):
        self.input = input
        self.to_train = to_train
        self.summary = []

    def Net(self):
        with tf.variable_scope("conv0"):
            W_1 = weight_variable([7, 7, 3, 64], "_1")
            conv2d_1 = tf.nn.conv2d(self.input, W_1, [1, 2, 2, 1], "SAME")
            self.summary.append(variable_summaries(W_1))

            bn_1 = tf.contrib.layers.batch_norm(conv2d_1, is_training=self.to_train)
            relu_1 = tf.nn.relu(bn_1)
            max_pool_1 = tf.nn.max_pool(relu_1, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

        conv1 = self.Res_Block(max_pool_1, 64, 64, 2, "1")
        conv2 = self.Res_Block(conv1, 64, 128, 2, "2")
        conv3 = self.Res_Block(conv2, 128, 256, 2, "3")
        conv4 = self.Res_Block(conv3, 256, 512, 2, "4")

        avg_pool_1 = tf.nn.avg_pool(conv4, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

        return avg_pool_1, self.summary

    def Res_Block(self, layer_input, input_channels, output_channels, block_num, no):
        input_ = layer_input
        with tf.variable_scope("conv%s" % no):
            for i in range(block_num):
                with tf.variable_scope("conv%s_%d" % (no, i)):
                    if i != 0:
                        W_1 = weight_variable([3, 3, output_channels, output_channels], "_1")
                        conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 1, 1, 1], "SAME")

                    else:
                        W_0 = weight_variable([1, 1, input_channels, output_channels], "_0")
                        W_1 = weight_variable([3, 3, input_channels, output_channels], "_1")
                        if no != "1":
                            conv2d_0 = tf.nn.conv2d(input_, W_0, [1, 2, 2, 1], "SAME")
                            bn_0 = tf.contrib.layers.batch_norm(conv2d_0, is_training=self.to_train)
                            conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 2, 2, 1], "SAME")
                        else:
                            conv2d_0 = tf.nn.conv2d(input_, W_0, [1, 1, 1, 1], "SAME")
                            bn_0 = tf.contrib.layers.batch_norm(conv2d_0, is_training=self.to_train)
                            conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 1, 1, 1], "SAME")

                    W_2 = weight_variable([3, 3, output_channels, output_channels], "_2")
                    self.summary += [variable_summaries(W_0), variable_summaries(W_1), variable_summaries(W_2)]

                    bn_1 = tf.contrib.layers.batch_norm(conv2d_1, is_training=self.to_train)
                    relu_1 = tf.nn.relu(bn_1)

                    conv2d_2 = tf.nn.conv2d(relu_1, W_2, [1, 1, 1, 1], "SAME")
                    bn_2 = tf.contrib.layers.batch_norm(conv2d_2, is_training=self.to_train)
                    relu_2 = tf.nn.relu(bn_2)

                    if i != 0:
                        add = tf.nn.relu(tf.add(relu_2, input_))
                    else:
                        add = tf.nn.relu(tf.add(relu_2, bn_0))

                    input_ = add

            return input_

class ResNet34_Depth:

    def __init__(self, input, to_train):
        self.input = input
        self.to_train = to_train
        self.summary = []

    def Net(self):
        with tf.variable_scope("conv0"):
            W_1 = weight_variable([7, 7, 1, 32], "_1")
            conv2d_1 = tf.nn.conv2d(self.input, W_1, [1, 2, 2, 1], "SAME")
            self.summary.append(variable_summaries(W_1))

            bn_1 = tf.contrib.layers.batch_norm(conv2d_1, is_training=self.to_train)
            relu_1 = tf.nn.relu(bn_1)
            max_pool_1 = tf.nn.max_pool(relu_1, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

        conv1 = self.Res_Block(max_pool_1, 32, 32, 2, "1")
        conv2 = self.Res_Block(conv1, 32, 64, 2, "2")
        conv3 = self.Res_Block(conv2, 64, 128, 2, "3")
        conv4 = self.Res_Block(conv3, 128, 256, 2, "4")

        avg_pool_1 = tf.nn.avg_pool(conv4, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

        return avg_pool_1, self.summary

    def Res_Block(self, layer_input, input_channels, output_channels, block_num, no):
        input_ = layer_input
        with tf.variable_scope("conv%s" % no):
            for i in range(block_num):
                with tf.variable_scope("conv%s_%d" % (no, i)):
                    if i != 0:
                        W_1 = weight_variable([3, 3, output_channels, output_channels], "_1")
                        conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 1, 1, 1], "SAME")

                    else:
                        W_0 = weight_variable([1, 1, input_channels, output_channels], "_0")
                        W_1 = weight_variable([3, 3, input_channels, output_channels], "_1")
                        if no != "1":
                            conv2d_0 = tf.nn.conv2d(input_, W_0, [1, 2, 2, 1], "SAME")
                            bn_0 = tf.contrib.layers.batch_norm(conv2d_0, is_training=self.to_train)
                            conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 2, 2, 1], "SAME")
                        else:
                            conv2d_0 = tf.nn.conv2d(input_, W_0, [1, 1, 1, 1], "SAME")
                            bn_0 = tf.contrib.layers.batch_norm(conv2d_0, is_training=self.to_train)
                            conv2d_1 = tf.nn.conv2d(input_, W_1, [1, 1, 1, 1], "SAME")

                    W_2 = weight_variable([3, 3, output_channels, output_channels], "_2")
                    self.summary += [variable_summaries(W_0), variable_summaries(W_1), variable_summaries(W_2)]

                    bn_1 = tf.contrib.layers.batch_norm(conv2d_1, is_training=self.to_train)
                    relu_1 = tf.nn.relu(bn_1)

                    conv2d_2 = tf.nn.conv2d(relu_1, W_2, [1, 1, 1, 1], "SAME")
                    bn_2 = tf.contrib.layers.batch_norm(conv2d_2, is_training=self.to_train)
                    relu_2 = tf.nn.relu(bn_2)

                    if i != 0:
                        add = tf.nn.relu(tf.add(relu_2, input_))
                    else:
                        add = tf.nn.relu(tf.add(relu_2, bn_0))

                    input_ = add

            return input_

# class ResNet34_Depth:
#
#     def __init__(self, input, to_train):
#         self.input = input
#         self.to_train = False
#
#     def Net(self):
#         with tf.variable_scope("conv0"):
#             W_1 = weight_variable([7, 7, 1, 32], "_1", to_train=True)
#             conv3d_1 = tf.nn.conv3d(self.input, W_1, [1, 1, 2, 2, 1], "SAME")
#
#             bn_1 = tf.contrib.layers.batch_norm(conv3d_1, is_training=self.to_train)
#             relu_1 = tf.nn.relu(bn_1)
#             max_pool_1 = tf.nn.max_pool3d(relu_1, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], "SAME")
#
#         conv1 = self.Res_Block(max_pool_1, 2, 32, 32, 2, "1")
#         conv2 = self.Res_Block(conv1, 2, 32, 64, 2, "2")
#         conv3 = self.Res_Block(conv2, 2, 64, 128, 2, "3")
#         conv4 = self.Res_Block(conv3, 2, 128, 256, 2, "4")
#
#         avg_pool_1 = tf.nn.avg_pool3d(conv4, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], "SAME")
#
#         return avg_pool_1
#
#     def Res_Block(self, layer_input, depth, input_channels, output_channels, block_num, no, to_train=True):
#         input_ = layer_input
#         with tf.variable_scope("conv%s" % no):
#             for i in range(block_num):
#                 with tf.variable_scope("conv%s_%d" % (no, i)):
#                     if i != 0:
#                         W_1 = weight_variable([depth, 3, 3, output_channels, output_channels], "_1", to_train=True)
#                         conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
#
#                     else:
#                         W_0 = weight_variable([depth, 1, 1, input_channels, output_channels], "_0", to_train=True)
#                         W_1 = weight_variable([depth, 3, 3, input_channels, output_channels], "_1", to_train=True)
#                         if no != "1":
#                             conv3d_0 = tf.nn.conv3d(input_, W_0, [1, 1, 2, 2, 1], "SAME")
#                             bn_0 = tf.contrib.layers.batch_norm(conv3d_0, is_training=self.to_train)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 2, 2, 1], "SAME")
#                         else:
#                             conv3d_0 = tf.nn.conv3d(input_, W_0, [1, 1, 1, 1, 1], "SAME")
#                             bn_0 = tf.contrib.layers.batch_norm(conv3d_0, is_training=self.to_train)
#                             conv3d_1 = tf.nn.conv3d(input_, W_1, [1, 1, 1, 1, 1], "SAME")
#
#                     W_2 = weight_variable([depth, 3, 3, output_channels, output_channels], "_2", to_train=True)
#
#                     bn_1 = tf.contrib.layers.batch_norm(conv3d_1, is_training=self.to_train)
#                     relu_1 = tf.nn.relu(bn_1)
#
#                     conv3d_2 = tf.nn.conv3d(relu_1, W_2, [1, 1, 1, 1, 1], "SAME")
#                     bn_2 = tf.contrib.layers.batch_norm(conv3d_2, is_training=self.to_train)
#                     relu_2 = tf.nn.relu(bn_2)
#
#                     if i != 0:
#                         add = tf.nn.relu(tf.add(relu_2, input_))
#                     else:
#                         add = tf.nn.relu(tf.add(relu_2, bn_0))
# 
#                     input_ = add
# 
#             return input_