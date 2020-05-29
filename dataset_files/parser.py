import numpy as np
import scipy.misc as smc
from natsort import natsorted as ns
import glob, os
import argparse

# parser = argparse.ArgumentParser(description="Create Lidar Dataset Parser file")
# parser.add_argument("path", help = "path_to_folder", type = str)
# args = parser.parse_args()

# dataset_path = args.path
dataset_path = "./dataset_files/dataset/2011_09_26/"
time_step = 3

#Picking up all sync folders
folder_names = ns(glob.glob(dataset_path +"*_sync" + os.path.sep))

dataset_array = np.zeros(dtype = str, shape = (1,20))
dataset_array_2 = np.zeros(dtype = str, shape = (1,20))

def padding(array, time_step):
    r, c = array.shape
    group = np.floor(r/time_step)
    left = int(r - time_step * group)
    if left >= 1:
        return np.vstack((array[:-1 - left], array[-1-time_step:]))
    else:
        return array[:r - left]


for fn in folder_names:
    print(fn)
    file_names_source = ns(glob.glob(fn + "depth_maps_transformed_ite/*.png"))
    size = np.shape(file_names_source)[0]
    file_names_target = ns(glob.glob(fn + "depth_maps/*.png"))[:size]
    img_source = ns(glob.glob(fn + "image_02/data/*.png"))[:size]
    img_target = ns(glob.glob(fn + "image_03/data/*.png"))[:size]
    transforms_list = np.loadtxt(fn + "angle_list_ite.txt", dtype = str)[:size]

    file_names_source = np.array(file_names_source, dtype=str).reshape(-1,1)
    file_names_target = np.array(file_names_target, dtype=str).reshape(-1,1)
    img_source = np.array(img_source, dtype=str).reshape(-1,1)
    img_target = np.array(img_target, dtype=str).reshape(-1,1)

    dataset = np.hstack((file_names_source, file_names_target, img_source, img_target, transforms_list))
    # dataset = padding(dataset, time_step)
    print(dataset.shape)

    dataset_array = np.vstack((dataset_array, dataset))

    #######################################################################################

    file_names_source_2 = ns(glob.glob(fn + "depth_maps_transformed_ite_2/*.png"))[:size]
    file_names_target_2 = ns(glob.glob(fn + "depth_maps_2/*.png"))[:size]

    transforms_list_2 = np.loadtxt(fn + "angle_list_ite_2.txt", dtype = str)[:size]

    file_names_source_2 = np.array(file_names_source_2, dtype=str).reshape(-1,1)
    file_names_target_2 = np.array(file_names_target_2, dtype=str).reshape(-1,1)

    dataset_2 = np.hstack((file_names_source_2, file_names_target_2, img_source, img_target, transforms_list_2))
    # dataset_2 = padding(dataset_2, time_step)
    print(dataset_2.shape)

    dataset_array_2 = np.vstack((dataset_array_2, dataset_2))




dataset_array = dataset_array[1:]
dataset_array_2 = dataset_array_2[1:]

final_array = np.vstack((dataset_array, dataset_array_2))

print(final_array.shape)
r = final_array.shape[0]
print(final_array.shape)
final_array = np.reshape(final_array, (int(r / time_step), time_step, 20))
print(final_array.shape)
np.random.shuffle(final_array)
final_array = np.reshape(final_array, (r, 20))

# train = int(24000 / time_step)
# val = int(3000 / time_step)
# test = int(3000 / time_step)
# np.random.shuffle(final_array[0:train])
# np.random.shuffle(final_array[train:train + val])
# np.random.shuffle(final_array[train + val:])

np.savetxt("./dataset_files/parsed_set.txt", final_array[:29800], fmt = "%s", delimiter=' ')
