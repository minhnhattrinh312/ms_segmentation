import nibabel as nib
import glob
import os
import numpy as np
from tqdm import tqdm
from segment3d import load_yaml, getValidIndex, z_score_normalize, extract_patches, config, min_max_normalize
import csv
import random

def make_data2npz(list_file_flair, data_mean_std, save_path="./data_np/"):
    os.makedirs(save_path, exist_ok=True)  
    for count_subject, path_flair in tqdm(enumerate(list_file_flair, 1)):
        id = path_flair.split("/")[-1]
        parts = id.split("_")
        # Access the first two elements in the list (index 0 and 1)
        id = parts[0] + "_" + parts[1]
        name_subject = f"mri_{count_subject}"
        fold = int(parts[0][-2:])
        
        flair = nib.load(path_flair).get_fdata()
        mprage = nib.load(path_flair.replace("flair", "mprage")).get_fdata()
        # pd = nib.load(path.replace("flair", "pd")).get_fdata()
        # t2 = nib.load(path.replace("flair", "t2")).get_fdata()
        mask1 = nib.load(path_flair.replace("flair_pp", "mask1").replace("preprocessed", "masks")).get_fdata()[np.newaxis]
        mask2 = nib.load(path_flair.replace("flair_pp", "mask2").replace("preprocessed", "masks")).get_fdata()[np.newaxis]
        mask = np.logical_or(mask1, mask2).astype(np.int64)
        
        validIndex = (slice(None),) + getValidIndex(flair, size=config["DATA"]["CROP_SIZE"])


        flair = z_score_normalize(flair, data_mean_std["mean_flair"], data_mean_std["std_flair"])
        mprage = z_score_normalize(mprage, data_mean_std["mean_mprage"], data_mean_std["std_mprage"])
        # pd = z_score_normalize(pd, data_mean_std["mean_pd"], data_mean_std["std_pd"])
        # t2 = z_score_normalize(t2, data_mean_std["mean_t2"], data_mean_std["std_t2"])

        # inputs = np.stack([flair, mprage, pd, t2], axis=0)
        inputs = np.stack([flair, mprage], axis=0)
        inputs = np.asarray(inputs[validIndex], np.float32)
        mask = np.asarray(mask[validIndex], np.int64)

        np.savez_compressed(f"{save_path}{name_subject}", image=inputs, mask=mask)
        
        with open(f"subject2name.csv", 'a', newline='') as csvfile:
            # Check if the file is empty
            if csvfile.tell() == 0:
                # If the file is empty, write the header row
                writer = csv.DictWriter(csvfile, fieldnames=["subject", "name", "fold"])
                writer.writeheader()
            else:
                # If the file is not empty, create a writer without a header row
                writer = csv.DictWriter(csvfile, fieldnames=["subject", "name", "fold"], lineterminator='\n')
            # Write the new id and score to the file
            writer.writerow({"subject": id, "name": name_subject, "fold": fold})
            
def extract_data2npz(list_file_flair, config, data_mean_std, distinct_subject=False, save_path="./data_np/"):
    os.makedirs(save_path, exist_ok=True)
    rows = []
    num_fold = 10
    # Calculate the frequency of each number
    frequency = len(list_file_flair) // num_fold
    # Create the list of folds
    folds = []
    for i in range(1, num_fold+1):
        folds += [i] * frequency
    # Add the remaining elements to the list
    folds += [i + 1 for i in range(len(list_file_flair) - len(folds))]
    random.shuffle(folds)
    
    print("use: ", config["TRAIN"]["MASK"])
    print("distinct_subject: ", distinct_subject)
    useless = 0
    num_zero_list, num_non_zero_list = [], []
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
        mprage = nib.load(path_flair.replace("flair", "mprage")).get_fdata()
        # pd = nib.load(path_flair.replace("flair", "pd")).get_fdata()
        t2 = nib.load(path_flair.replace("flair", "t2")).get_fdata()
        
        mask1 = nib.load(path_flair.replace("flair_pp", "mask1").replace("preprocessed", "masks")).get_fdata()[np.newaxis]
        mask2 = nib.load(path_flair.replace("flair_pp", "mask2").replace("preprocessed", "masks")).get_fdata()[np.newaxis]
        if config["TRAIN"]["MASK"] == "mask2":
            mask = mask2.astype(np.int64)
        elif config["TRAIN"]["MASK"] == "mask1":
            mask = mask1.astype(np.int64)
        else:
            mask = np.logical_or(mask1, mask2).astype(np.int64)

        validIndex = (slice(None),) + getValidIndex(flair, size=config["DATA"]["CROP_SIZE"])
        
        if config["TRAIN"]["NORMALIZE"] == "z_score":
            flair = z_score_normalize(flair, data_mean_std["mean_flair"], data_mean_std["std_flair"])
            mprage = z_score_normalize(mprage, data_mean_std["mean_mprage"], data_mean_std["std_mprage"])
            # pd = z_score_normalize(pd, data_mean_std["mean_pd"], data_mean_std["std_pd"])
            t2 = z_score_normalize(t2, data_mean_std["mean_t2"], data_mean_std["std_t2"])
        elif config["TRAIN"]["NORMALIZE"] == "min_max":
            flair = min_max_normalize(flair)
            mprage = min_max_normalize(mprage)
            # pd = min_max_normalize(pd)
            t2 = min_max_normalize(t2)
        else:
            raise ValueError("Invalid normalization method")
        
        # inputs = np.stack([flair, mprage, pd, t2], axis=0)
        # inputs = np.stack([flair, mprage], axis=0)
        inputs = np.stack([flair, mprage, t2], axis=0)
        
        inputs = np.asarray(inputs[validIndex], np.float32)
        mask = np.asarray(mask[validIndex], np.int64)

        patches = extract_patches(inputs, patch_shape=config["DATA"]["PATCH_SIZE"], extraction_step=config["DATA"]["EXTRACTION_STEP"])
        patches_mask = extract_patches(mask, patch_shape=config["DATA"]["PATCH_MASK"], extraction_step=config["DATA"]["EXTRACTION_STEP_MASK"])

        for i in range(patches.shape[0]):
            if np.sum(patches_mask[i]) != 0 and patches[i] is not None:
                _, (num_zero, num_non_zero) = np.unique(patches_mask[i], return_counts=True)
                num_zero_list.append(num_zero)
                num_non_zero_list.append(num_non_zero)
                rows.append({"subject": id, "name": f"mri_{count_subject}_{i}", "fold": fold})
                np.savez_compressed(f"{save_path}mri_{count_subject}_{i}", image=patches[i], mask=patches_mask[i])
            else:
                useless += 1
                
    print("useless: ", useless)
    print("number of zeros: ", np.sum(num_zero_list))
    print("number of non-zeros: ", np.sum(num_non_zero_list))
    print("ratio: ", np.sum(num_non_zero_list) / np.sum(num_zero_list))
    
    with open(f"subject2name.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["subject", "name", "fold"], lineterminator='\n')
        writer.writeheader()
        # Write the new id and score to the  file
        writer.writerows(rows)
            
if __name__ == "__main__":
    list_file_flair = sorted(glob.glob("data/data_isbi_2015/training/*/preprocessed/*flair*"))
    data_mean_std = load_yaml("mean_std_isbi2015.yaml")
    # make_data2npz(list_file_flair, data_mean_std, save_path="data/data_isbi_2015/isbi2npz3D/")
    extract_data2npz(list_file_flair, config, data_mean_std, \
        distinct_subject=config["TRAIN"]["DISTINCT_SUBJECT"], save_path="data/data_isbi_2015/isbi2npz3D/")