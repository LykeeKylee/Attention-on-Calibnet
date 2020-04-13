import numpy as np
import glob, os, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv
import config_res as config

total = config.net_params['total_frames']
total_train = config.net_params['total_frames_train']
total_validation = config.net_params['total_frames_validation']
total_test = config.net_params['total_frames_test']
partition_limit = config.net_params['partition_limit']

IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
batch_size = config.net_params['batch_size']
time_step = config.net_params['time_step']

dataset = np.loadtxt(config.paths['dataset_path_full'], dtype = str)

# 训练部分
dataset_train = dataset[:total_train]
# 验证部分
dataset_validation = dataset[total_train:total_train + total_validation]
# 测试部分
dataset_test = dataset[total_train + total_validation:total]

# 打乱
def shuffle(train, validation):
    train = np.reshape(train, (int(total_train / time_step), time_step, 20))
    validation = np.reshape(validation, (int(total_validation / time_step), time_step, 20))
    np.random.shuffle(train)
    np.random.shuffle(validation)
    train = np.reshape(train, (total_train, 20))
    validation = np.reshape(validation, (total_validation, 20))
    return train, validation


def load(p_no, mode):

    if(mode == "train"):
        dataset_part = dataset_train[p_no*partition_limit:(p_no + 1)*partition_limit]
    elif(mode == "validation"):
        dataset_part = dataset_validation[p_no*partition_limit:(p_no + 1)*partition_limit]
    elif mode=='test':
        dataset_part = dataset_test[p_no*partition_limit:(p_no+1)*partition_limit]

    source_file_names = dataset_part[:,0]
    target_file_names = dataset_part[:,1]
    source_image_names = dataset_part[:,2]
    target_image_names = dataset_part[:,3]
    transforms = np.float32(dataset_part[:,4:])

    target_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 1), dtype = np.float32)
    source_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 1), dtype = np.float32)
    source_img_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 3), dtype = np.float32)
    target_img_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 3), dtype = np.float32)
    transforms_container = np.zeros((partition_limit, 4, 4), dtype = np.float32)

    c_idx = 0
    for s_name, t_name, img_source_name, img_target_name, transform in tqdm(zip(source_file_names, target_file_names, source_image_names, target_image_names, transforms)):

        warped_ip = np.float32(cv.imread(s_name, flags=0))
        warped_ip[0:5,:] = 0.0 ; warped_ip[:,0:5] = 0.0 ; warped_ip[IMG_HT - 5:,:] = 0.0 ; warped_ip[:,IMG_WDT-5:] = 0.0
        warped_ip = (warped_ip - 40.0)/40.0
        # print(warped_ip.shape)
        source_container[c_idx, :, :, 0] = warped_ip

        target_ip = np.float32(cv.imread(t_name, flags=0))
        target_ip[0:5,:] = 0.0 ; target_ip[:,0:5] = 0.0 ; target_ip[IMG_HT - 5:,:] = 0.0 ; target_ip[:,IMG_WDT-5:] = 0.0
        target_ip = (target_ip - 40.0)/40.0
        target_container[c_idx, :, :, 0] = target_ip

        source_img = np.float32(cv.imread(img_source_name))
        source_img[0:5,:,:] = 0.0 ; source_img[:,0:5,:] = 0.0 ; source_img[IMG_HT - 5:,:,:] = 0.0 ; source_img[:,IMG_WDT-5:,:] = 0.0
        source_img = (source_img - 127.5)/127.5
        source_img_container[c_idx, :, :, :] = source_img

        target_img = np.float32(cv.imread(img_target_name))
        target_img[0:5,:,:] = 0.0 ; target_img[:,0:5,:] = 0.0 ; target_img[IMG_HT - 5:,:,:] = 0.0 ; target_img[:,IMG_WDT-5:,:] = 0.0
        target_img = (target_img - 127.5)/127.5
        target_img_container[c_idx, :, :, :] = target_img

        transforms_container[c_idx, :, :] = np.linalg.inv(transform.reshape(4,4))
        c_idx+=1

    source_container = source_container.reshape(int(partition_limit/batch_size/time_step), batch_size * time_step, IMG_HT, IMG_WDT , 1)
    target_container = target_container.reshape(int(partition_limit/batch_size/time_step), batch_size * time_step, IMG_HT, IMG_WDT , 1)
    source_img_container = source_img_container.reshape(int(partition_limit/batch_size/time_step), batch_size * time_step, IMG_HT, IMG_WDT, 3)
    target_img_container = target_img_container.reshape(int(partition_limit/batch_size/time_step), batch_size * time_step, IMG_HT, IMG_WDT, 3)
    transforms_container = transforms_container.reshape(int(partition_limit/batch_size/time_step), batch_size * time_step, 4, 4)

    return source_container, target_container, source_img_container, target_img_container, transforms_container
