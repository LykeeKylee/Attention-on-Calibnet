import numpy as np
import cv2 as cv
import common.Lie_functions
import tensorflow as tf

def dmap_rgb(source_rgb, dmap_exp):
    source_rgb = source_rgb * 127.5 + 127.5
    dmap_exp = dmap_exp / np.max(dmap_exp) * 255

    source_rgb = np.array(source_rgb, np.uint8)
    dmap_exp = np.array(dmap_exp, np.uint8)

    dmap_rgb = cv.applyColorMap(dmap_exp, cv.COLORMAP_HOT)
    mix = cv.addWeighted(source_rgb, 0.7, dmap_rgb, 1, 0)
    return mix


def rot2euler(rot):
    theta_x = np.arctan2(rot[2][1], rot[2][2])
    theta_y = np.arctan2(-rot[2][1], np.sqrt(rot[2][1]**2 + rot[2][2]**2))
    theta_z = np.arctan2(rot[1][0], rot[0][0])
    return np.array([theta_z, theta_y, theta_x])


def convert(matrix_T):
    vec_tr = matrix_T[:3, 3]
    vec_tr = tf.reshape(vec_tr, (1, 3))

    matrix_ro = matrix_T[:3, :3]
    thetax = tf.atan2(matrix_ro[2:3, 1:2], matrix_ro[2:3, 2:3])
    thetay = tf.atan2(-matrix_ro[2:3, 1:2], tf.sqrt(matrix_ro[2:3, 1:2] * matrix_ro[2:3, 1:2] + matrix_ro[2:3, 2:3] * matrix_ro[2:3, 2:3]))
    thetaz = tf.atan2(matrix_ro[1:2, 0:1], matrix_ro[0:1, 0:1])
    vec_ro = tf.concat([thetax, thetay, thetaz], axis=1)
    return tf.concat([vec_tr, vec_ro], axis=1)

def contrast(transforms_pred, transforms_exp):
    tran_pred = transforms_pred[:3, 3]
    tran_exp = transforms_exp[:3, 3]

    euler_angle_pred = rot2euler(transforms_pred[0:3, 0:3])
    euler_angle_exp = rot2euler(transforms_exp[0:3, 0:3])

    return np.abs(euler_angle_pred - euler_angle_exp) /np.pi * 180, (np.abs(tran_pred - tran_exp)) * 100

def RV2RM(tran_matrix, v):
    rot_matrix = tran_matrix[:3, :3]
    A = tf.subtract(rot_matrix, tf.transpose(rot_matrix)) / 2
    theta = tf.acos((tf.trace(rot_matrix) - 1) / 2)
    v_u = tf.divide(A, tf.sin(theta))
    B = tf.sin(theta) / theta
    j =  tf.eye(3) + (1 - B) * tf.matmul(v_u, v_u) + (1 - tf.cos(theta)) / theta * v_u
    nv = tf.matmul(tf.reshape(v[3:],shape=[1, 3]), j)
    nv = tf.concat([tf.reshape(v[:3],shape=[1, 3]), nv], axis=1)
    return nv