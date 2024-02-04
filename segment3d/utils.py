import numpy as np
import torch
import yaml
from itertools import product

def load_yaml(path_config):
    with open(path_config, 'r') as fp:
        cfg = yaml.safe_load(fp)
    return cfg

def extract_patches(arr, patch_shape, extraction_step):
    arr_ndim = arr.ndim
    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    indexing_strides = arr[tuple(slices)].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1
    
    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    patches = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    npatches = np.prod(patches.shape[:arr_ndim])
    return patches.reshape([npatches, ] + patch_shape)

def reconstruct_volume_avg(patches, expected_shape, extraction_step, num_class = 2):
    v_x, v_y, v_z = expected_shape
    p_x, p_y, p_z = patches.shape[2:]
    s_x, s_y, s_z = extraction_step[1:]# compute the dimensions of the patches array
    n_x = (v_x - p_x) // s_x + 1
    n_y = (v_y - p_y) // s_y + 1
    n_z = (v_z - p_z) // s_z + 1

    vol = np.zeros((num_class, v_x, v_y, v_z))
    count = np.zeros((num_class, v_x, v_y, v_z))

    for p, (i, j, k) in zip(patches, product(range(n_x), range(n_y), range(n_z))):
        vol[:, i*s_x:i*s_x+ p_x, j*s_y:j*s_y + p_y, k*s_z:k*s_z + p_z]  +=p
        count[:, i*s_x:i*s_x+ p_x, j*s_y:j*s_y + p_y, k*s_z:k*s_z + p_z] +=1
    return vol/count

def getValidIndex(volume, size=(160, 208, 160)):
    """ this function remove unnescessary background and center crop remaining part """
    x_indexes, y_indexes, z_indexes = np.nonzero(volume)
    dims_min = np.min([x_indexes, y_indexes, z_indexes], axis=1)
    dims_max = np.max([x_indexes, y_indexes, z_indexes], axis=1)

    dims_min = dims_min + (dims_max - dims_min - size) // 2
    dims_min[dims_min < 0] = 0
    # dims_min[-1] = 0 # because size[-1] always is larger than true size, remove this line for another task.
    dims_max = dims_min + size
    return tuple(slice(imin, imax) for imin, imax in zip(dims_min, dims_max))

def z_score_normalize(volume, mean, std):
    return (volume - mean) / std

def min_max_normalize(volume):
    volume = (volume - np.min(volume)) / (np.max(volume)-np.min(volume))
    return volume