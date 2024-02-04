import sys
import numpy as np
from tqdm import tqdm
import os
import glob
import nibabel as nib
sys.path.insert(0, '/home/nhattm1/test_newcode/')
from segment2d import min_max_normalize, cfg, pad_background

list_file_flair = sorted(glob.glob("./data/brats2021/*/*flair*"))

def extract_brats2npz(list_file_flair, cfg, save_path="./data_np/"):
    os.makedirs(save_path, exist_ok=True)
    
    useless = 0
    views = ["coronal", "sagittal", "axial"]
    print(f"use ", cfg.TRAIN.NORMALIZE, " normalization")
    for count_subject, path_flair in tqdm(enumerate(list_file_flair, 1)):

        flair = nib.load(path_flair).get_fdata()
        t1 = nib.load(path_flair.replace("flair", "t1")).get_fdata()
        t2 = nib.load(path_flair.replace("flair", "t2")).get_fdata()

        padded_flair, _, _ = pad_background(flair, dim2pad=cfg.DATA.DIM2PAD_ISBI)
        padded_t1, _, _ = pad_background(t1, dim2pad=cfg.DATA.DIM2PAD_ISBI)
        padded_t2, _, _ = pad_background(t2, dim2pad=cfg.DATA.DIM2PAD_ISBI)

        padded_flair = min_max_normalize(padded_flair)
        padded_t1 = min_max_normalize(padded_t1)
        padded_t2 = min_max_normalize(padded_t2)


        for view, transpose_view in zip(views, cfg.DATA.ORIGIN2CUT):
            transposed_flair = np.transpose(padded_flair, transpose_view)
            transposed_t1 = np.transpose(padded_t1, transpose_view)
            transposed_t2 = np.transpose(padded_t2, transpose_view)
            
            for i in range(transposed_flair.shape[-1]):
                slices_flair = transposed_flair[..., i] # shape (224, 224, 1)
                slices_t1 = transposed_t1[..., i] # shape (224, 224, 1)
                slices_t2 = transposed_t2[..., i] # shape (224, 224, 1)
                slice_inputs = np.stack([slices_t1, slices_flair, slices_t2], axis=-1) #shape (224, 224, 3)

                if np.count_nonzero(slices_flair) >= 2:
                    name_subject = f"brats{count_subject}_{view}_{i}"
                    np.savez_compressed(f"{save_path}{name_subject}", MRimages=slice_inputs)
                else:
                    useless += 1
                    
                
    print("useless: ", useless)
    
if __name__ == "__main__":
    list_file_flair = sorted(glob.glob("./data/brats/*/*/*flair*"))
    extract_brats2npz(list_file_flair, cfg, save_path="./data/brats/brats_npz/")