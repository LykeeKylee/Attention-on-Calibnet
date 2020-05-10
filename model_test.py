import numpy as np
import tensorflow as tf
import cv2 as cv

import config_res as config

from common.cnn_utils_res import *
from common import all_transformer as at3
from common import global_agg_net
from common.Lie_functions import exponential_map_single

import nw_loader_color as ldr
import model_utils
import time
import transform_functions


_ALPHA_CONST = 1.0
_BETA_CONST = 1.0
IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
batch_size = config.net_params['batch_size']
learning_rate = config.net_params['learning_rate']
n_epochs = config.net_params['epochs']
current_epoch = config.net_params['load_epoch']

# 重置默认图
tf.reset_default_graph()

X1 = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 3), name = "X1")
X2 = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 1), name = "X2")
# TODO 两张连续的图层叠

depth_maps_target = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 1), name = "depth_maps_target")
expected_transforms = tf.placeholder(tf.float32, shape = (batch_size, 4, 4), name = "expected_transforms")

phase = tf.placeholder(tf.bool, [], name = "phase")
phase_rgb = tf.placeholder(tf.bool, [], name = "phase_rgb")
fc_keep_prob = tf.placeholder(tf.float32, name = "fc_keep_prob")

fx = config.camera_params['fx']
fy = config.camera_params['fy']
cx = config.camera_params['cx']
cy = config.camera_params['cy']

fx_scaled = 2*(fx)/np.float32(IMG_WDT)              # focal length x scaled for -1 to 1 range
fy_scaled = 2*(fy)/np.float32(IMG_HT)               # focal length y scaled for -1 to 1 range
cx_scaled = -1 + 2*(cx - 1.0)/np.float32(IMG_WDT)   # optical center x scaled for -1 to 1 range
cy_scaled = -1 + 2*(cy - 1.0)/np.float32(IMG_HT)    # optical center y scaled for -1 to 1 range

K_mat_scaled = np.array([[fx_scaled,  0.0, cx_scaled],
                         [0.0, fy_scaled,  cy_scaled],
                         [0.0, 0.0, 1.0]], dtype = np.float32)

K_final = tf.constant(K_mat_scaled, dtype = tf.float32)
small_transform = tf.constant(config.camera_params['cam_transform_02_inv'], dtype = tf.float32)

X2_pooled = tf.nn.max_pool(X2, ksize=[1,5,5,1], strides=[1,1,1,1], padding="SAME")
# X2_pooled = X2

depth_maps_target_pooled = tf.nn.max_pool(depth_maps_target, ksize=[1,5,5,1], strides=[1,1,1,1], padding="SAME")
# depth_maps_target_pooled = depth_maps_target

# output_vectors, weight_summaries = global_agg_net.End_Net_Out(X1, phase_rgb, X2_pooled, phase, keep_prob)
net = global_agg_net.Nets(X1, X2_pooled, phase_rgb, phase, fc_keep_prob)
output_vectors, weight_summaries = net.build()

# se(3) -> SE(3) for the whole batch
# output_vectors_ft = tf.map_fn(lambda x:dmap.RV2RM(expected_transforms[x], output_vectors[x]), elems=tf.range(0, batch_size, 1), dtype=tf.float32)
# output_vectors_ft = tf.reshape(output_vectors_ft, shape=(batch_size, 6))
predicted_transforms = tf.map_fn(lambda x:exponential_map_single(output_vectors[x]), elems=tf.range(0, batch_size, 1), dtype=tf.float32)

# predicted_transforms = tf.concat([expected_transforms[:, :3, :3], tf.reshape(output_vectors_ft[:, 3:], shape=[batch_size, 3, 1])], axis=-1)

# transforms depth maps by the predicted transformation
depth_maps_predicted, cloud_pred = tf.map_fn(lambda x:at3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, predicted_transforms[x], K_final, small_transform), elems = tf.range(0, batch_size, 1), dtype = (tf.float32, tf.float32))

# transforms depth maps by the expected transformation
depth_maps_expected, cloud_exp = tf.map_fn(lambda x:at3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, expected_transforms[x], K_final, small_transform), elems = tf.range(0, batch_size, 1), dtype = (tf.float32, tf.float32))

# photometric loss between predicted and expected transformation
photometric_loss = tf.nn.l2_loss(tf.subtract((depth_maps_expected[:,10:-10,10:-10] - 40.0)/40.0, (depth_maps_predicted[:,10:-10,10:-10] - 40.0)/40.0))

# earth mover's distance between point clouds
cloud_loss = model_utils.get_emd_loss(cloud_pred, cloud_exp)

# final loss term
train_loss = _ALPHA_CONST*photometric_loss + _BETA_CONST*cloud_loss

tf.add_to_collection('losses1', train_loss)
loss1 = tf.add_n(tf.get_collection('losses1'))


predicted_loss_test = tf.nn.l2_loss(tf.subtract((depth_maps_expected[:,10:-10,10:-10] - 40.0)/40.0, (depth_maps_predicted[:,10:-10,10:-10] - 40.0)/40.0))
cloud_loss_test = model_utils.get_emd_loss(cloud_pred, cloud_exp)
test_loss = _ALPHA_CONST * predicted_loss_test + _BETA_CONST * cloud_loss_test

test_summary_1 =  tf.summary.scalar('Test_photometric_loss', predicted_loss_test)
test_summary_2 = tf.summary.scalar('Test_cloud_loss', cloud_loss_test)
test_summary_3 = tf.summary.scalar('Test_loss', test_loss)

merge_test = tf.summary.merge([test_summary_1] + [test_summary_2] + [test_summary_3])

saver = tf.train.Saver(tf.all_variables())

# tensorflow gpu configuration. Not to be confused with network configuration file

config_tf = tf.ConfigProto(allow_soft_placement=True)
config_tf.gpu_options.allow_growth = True

with tf.Session(config = config_tf) as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./logs_simple_transformer/")

    total_iterations_train = 0
    total_iterations_validate = 0
    total_iterations_test = 0
    checkpoint_path = config.paths['checkpoint_path']

    total_partitions_test = config.net_params['total_frames_test']/config.net_params['partition_limit']

    ldr.dataset_train, ldr.dataset_validation = ldr.shuffle(ldr.dataset_train, ldr.dataset_validation)
    saver.restore(sess, checkpoint_path + "/model-%d" % 22)
    y, p, r = [], [], []
    x, y_, z = [], [], []
    for part in range(int(total_partitions_test)):
        source_container, target_container, source_img_container, target_img_container, transforms_container = ldr.load(part, mode = "test")

        for source_b, target_b, source_img_b, target_img_b, transforms_b in zip(source_container, target_container, source_img_container, target_img_container, transforms_container):

            outputs= sess.run([depth_maps_predicted, depth_maps_expected, predicted_loss_test, X2_pooled, merge_test, cloud_loss_test, predicted_transforms],
                              feed_dict={X1: source_img_b,
                                         X2: source_b,
                                         depth_maps_target: target_b,
                                         expected_transforms: transforms_b,
                                         phase: False,
                                         fc_keep_prob:1.0,
                                         phase_rgb: False})

            dmaps_pred = outputs[0]
            dmaps_exp = outputs[1]
            photo_loss = outputs[2]
            source = outputs[3]

            # writer.add_summary(outputs[4], total_iterations_test)
            total_iterations_test+=1

            print('Time: %s     Current Iteration of Test: %d' % (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()), total_iterations_test))
            print('Loss: %f' % (photo_loss * _ALPHA_CONST + outputs[5] * _BETA_CONST))
            print('Photometric Loss: %f %f\tCloud Loss: %f %f' % (photo_loss, _ALPHA_CONST * photo_loss, outputs[5], _BETA_CONST * outputs[5]))

            random_disp = np.random.randint(batch_size)
            yaw, pitch, roll = [], [], []
            X, Y, Z = [], [], []
            for disp in range(batch_size):
                dst = transform_functions.contrast(outputs[6][disp], transforms_b[disp])
                y.append(dst[0][0])
                p.append(dst[0][1])
                r.append(dst[0][2])
                x.append(dst[1][0])
                y_.append(dst[1][1])
                z.append(dst[1][2])
                yaw.append(dst[0][0])
                pitch.append(dst[0][1])
                roll.append(dst[0][2])
                X.append(dst[1][0])
                Y.append(dst[1][1])
                Z.append(dst[1][2])

            print('max_translation_error(X Y Z): %fcm %fcm %fcm' % (np.max(X), np.max(Y), np.max(Z)))
            print('max_rotation_error(yaw pitch roll): %f° %f° %f°' % (np.max(yaw), np.max(pitch), np.max(roll)))
            print('average_translation_error(X Y Z): %fcm %fcm %fcm' % (np.average(X), np.average(Y), np.average(Z)))
            print('average_rotation_error(yaw pitch roll): %f° %f° %f°' % (np.average(yaw), np.average(pitch), np.average(roll)))
            print()

            if(total_iterations_test%20 == 0):

                random_disp = np.random.randint(batch_size)
                dmap_rgb = transform_functions.dmap_rgb(source_img_b[random_disp], dmaps_pred[random_disp])
                dmap_rgb2 = transform_functions.dmap_rgb(source_img_b[random_disp], dmaps_exp[random_disp])

                cv.imwrite(config.paths['test_imgs_path'] + "/test_%d_save_%d.png"%(0, total_iterations_test), np.vstack((dmap_rgb, dmap_rgb2)))
    print('final_translation_error(X Y Z): %fcm %fcm %fcm %fcm' % (np.average(x), np.average(y_), np.average(z), (np.average(x) + np.average(y_) + np.average(z)) / 3))
    print('final_rotation_error(yaw pitch roll): %f° %f° %f° %f°' % (np.average(y), np.average(p), np.average(r), (np.average(y) + np.average(p) + np.average(r)) / 3))
