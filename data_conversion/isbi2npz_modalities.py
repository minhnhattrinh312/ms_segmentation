import nibabel as nib
import glob
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, "D:/visual_code/ms_segmentation/")
from segment2d import *
# from segment3d import load_yaml, getValidIndex, z_score_normalize, extract_patches, min_max_normalize, cfg, data_mean_std
import csv
import random

def extract2d_data2npz(list_file_flair, cfg, distinct_subject=False, save_path="./data_np/"):
    os.makedirs(save_path, exist_ok=True)
    rows = []
    num_fold = 5
    # Calculate the frequency of each number
    frequency = len(list_file_flair) // num_fold
    # Create the list of folds
    folds = []
    for i in range(1, num_fold+1):
        folds += [i] * frequency
    # Add the remaining elements to the list
    folds += [i + 1 for i in range(len(list_file_flair) - len(folds))]
    random.shuffle(folds)
    
    useless = 0
    num_zero_list, num_non_zero_list = [], []
    views = ["coronal", "sagittal", "axial"]
    print(f"use ", cfg.TRAIN.NORMALIZE, " normalization")
    for count_subject, path_flair in tqdm(enumerate(list_file_flair, 1)):
        id = path_flair.split("/")[-1]
        parts = id.split("_")
        # Access the first two elements in the list (index 0 and 1)
        id = parts[0] + "_" + parts[1]
        if distinct_subject:
            fold = int(parts[0][-2:])
        else:
            fold = folds.pop()
        
        flair = nib.load(path_flair).get_fdata()
        t1 = nib.load(path_flair.replace("flair", "mprage")).get_fdata()
        pd = nib.load(path_flair.replace("flair", "pd")).get_fdata()
        t2 = nib.load(path_flair.replace("flair", "t2")).get_fdata()
        
        flair = min_max_normalize(flair)
        t1 = min_max_normalize(t1)
        pd = min_max_normalize(pd)
        t2 = min_max_normalize(t2)
        
        mask1 = nib.load(path_flair.replace("flair_pp", "mask1").replace("preprocessed", "masks"))
        mask2 = nib.load(path_flair.replace("flair_pp", "mask2").replace("preprocessed", "masks"))
        
        mask1 = mask1.get_fdata().astype(np.uint8)
        mask2 = mask2.get_fdata().astype(np.uint8)

        padded_flair, crop_index, padded_index = pad_background(flair, dim2pad=cfg.DATA.DIM2PAD_ISBI)
        padded_t1 = pad_background_with_index(t1, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI)
        padded_pd = pad_background_with_index(pd, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI)
        padded_t2 = pad_background_with_index(t2, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI)
        padded_mask1 = pad_background_with_index(mask1, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI)
        padded_mask2 = pad_background_with_index(mask2, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI)
        
        padded_masks = [padded_mask1, padded_mask2]
        

    
        for padded_mask in padded_masks:
            _, (num_zero, num_non_zero) = np.unique(padded_mask, return_counts=True)
            num_zero_list.append(num_zero)
            num_non_zero_list.append(num_non_zero)
            
        for view, transpose_view in zip(views, cfg.DATA.ORIGIN2CUT):
            transposed_flair = np.transpose(padded_flair, transpose_view)
            transposed_t1 = np.transpose(padded_t1, transpose_view)
            transposed_t2 = np.transpose(padded_t2, transpose_view)
            transposed_pd = np.transpose(padded_pd, transpose_view)
            transposed_masks = [np.transpose(mask, transpose_view) for mask in padded_masks]
            
            for i in range(transposed_flair.shape[-1]):
                slices_flair = transposed_flair[..., i] # shape (224, 224, 3)
                slices_t1 = transposed_t1[..., i] # shape (224, 224, 3)
                slices_t2 = transposed_t2[..., i] # shape (224, 224, 3)
                slices_pd = transposed_pd[..., i] # shape (224, 224, 3)
                slice_inputs = np.stack([slices_t1, slices_flair, slices_t2, slices_pd], axis=-1) #shape (224, 224, 4)
                for mask_id, transposed_mask in enumerate(transposed_masks, 1):
                    slices_mask = transposed_mask[..., i] # shape (224, 224)
                    if np.count_nonzero(slices_mask) >= 2:
                        name_subject = f"mri{count_subject}_{view}_{i}_mask{mask_id}"
                        rows.append({"subject": id, "name": name_subject, "fold": fold})
                        np.savez_compressed(f"{save_path}{name_subject}", \
                                    flair=slice_inputs.astype(np.float32), mask=slices_mask)
                    else:
                        useless += 1
                    
                
    print("useless: ", useless)
    print("number of zeros: ", np.sum(num_zero_list))
    print("number of non-zeros: ", np.sum(num_non_zero_list))
    print("ratio: ", np.sum(num_non_zero_list) / np.sum(num_zero_list))
    
    with open(f"subject_isbi.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["subject", "name", "fold"], lineterminator='\n')
        writer.writeheader()
        # Write the new id and score to the  file
        writer.writerows(rows)
        
if __name__ == "__main__":
    list_file_flair = sorted(glob.glob("./data/data_isbi_2015/training/*/preprocessed/*flair*"))
    # make_data2npz(list_file_flair, data_mean_std, save_path="data/data_isbi_2015/isbi2npz3D/")
    # extract3d_data2npz(list_file_flair, cfg, data_mean_std, \
    #     distinct_subject=cfg.TRAIN.DISTINCT_SUBJECT, save_path="data/data_isbi_2015/isbi2npz3D/")
    extract2d_data2npz(list_file_flair, cfg, distinct_subject=cfg.TRAIN.DISTINCT_SUBJECT, save_path="./data/data_isbi_2015/isbi2npz2D/")