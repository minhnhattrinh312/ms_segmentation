import nibabel as nib
from segment3d import *
import torch
import numpy as np
import gc
import glob
import os
from tqdm import tqdm
import csv
import shutil
from skimage.morphology import remove_small_objects
from skimage.measure import label

# build a class to predict subjects to a folder
class Predict2Folder():
    def __init__(self, segmentor, list_file_flair, config, data_mean_std, \
                device, num_class=2, team_name="nhatvbd", mask_exist=True, \
                batch_size=4, min_size_remove=100, make_zip=True, name_archive="testdata1"):
        self.config = config
        self.data_mean_std = data_mean_std
        self.device = device
        self.num_class = num_class
        self.segmentor = segmentor
        self.list_file_flair = list_file_flair
        self.team_name = team_name
        self.mask_exist = mask_exist
        self.batch_size = batch_size
        self.min_size_remove = min_size_remove
        self.name_archive = name_archive
        self.zip = make_zip
        
    def predict_patches(self, images):
        """return the patches"""
        images = torch.from_numpy(images)
        y_preds = torch.zeros((images.size(0), self.num_class, images.size(2), images.size(3), images.size(4)), device= self.device)
        batch_start = 0
        batch_end = self.batch_size
        while batch_start < images.size(0):
            image = images[batch_start : batch_end]
            with torch.inference_mode():
                image = image.to(self.device)
                y_pred = self.segmentor(image)
                y_preds[batch_start : batch_end] = y_pred
            batch_start += self.batch_size
            batch_end += self.batch_size
        res = y_preds.cpu().numpy()
        del y_preds
        return res

    def remove_small_elements(self, segmentation_mask):
        # Convert segmentation mask values greater than 0 to 1
        pred_mask = segmentation_mask > 0
        # Remove small objects (connected components) from the binary image
        mask = remove_small_objects(pred_mask, min_size=self.min_size_remove)
        # Multiply original segmentation mask with the mask to remove small objects
        clean_segmentation = segmentation_mask * mask
        return clean_segmentation
    
    def predict_subject(self, path_flair):
        flair = nib.load(path_flair).get_fdata()
        mprage = nib.load(path_flair.replace("flair", "mprage")).get_fdata()
        # pd = nib.load(path_flair.replace("path_flair", "pd")).get_fdata()
        t2 = nib.load(path_flair.replace("path_flair", "t2")).get_fdata()

        validIndex = (slice(None),) + getValidIndex(flair, size=self.config["DATA"]["CROP_SIZE"])

        flair = z_score_normalize(flair, self.data_mean_std["mean_flair"], self.data_mean_std["std_flair"])
        mprage = z_score_normalize(mprage, self.data_mean_std["mean_mprage"], self.data_mean_std["std_mprage"])
        # pd = z_score_normalize(pd, self.data_mean_std["mean_pd"], self.data_mean_std["std_pd"])
        t2 = z_score_normalize(t2, self.data_mean_std["mean_t2"], self.data_mean_std["std_t2"])

        # inputs = np.stack([flair, mprage, pd, t2], axis=0)
        # inputs = np.stack([flair, mprage], axis=0)
        inputs = np.stack([flair, mprage, t2], axis=0)
        
        inputs = np.asarray(inputs[validIndex], np.float32)

        patches = extract_patches(inputs, patch_shape=self.config["DATA"]["PATCH_SIZE"], extraction_step=self.config["DATA"]["EXTRACTION_STEP"])
        pred_patch = self.predict_patches(patches)
        gc.collect()
        torch.cuda.empty_cache()
        probability_output = reconstruct_volume_avg(pred_patch, inputs.shape[1:], self.config["DATA"]["EXTRACTION_STEP"])
        

        final_prediction = np.argmax(probability_output, axis=0)

        seg = np.zeros_like(flair)
        seg[validIndex[1:]] = final_prediction
        
        # post-processing by removing small connected components
        seg = self.remove_small_elements(seg)
        
        return seg

    def submit2folder(self):
        save_dir = self.config["DIRS"]["PREDICT_DIR"]
        rows_save = []
        for path_flair in tqdm(self.list_file_flair):
            id = path_flair.split("/")[-1]
            parts = id.split("_")
            # Access the first two elements in the list (index 0 and 1)
            id = parts[0] + "_" + parts[1]
            fold = int(parts[0][-2:])

            seg = self.predict_subject(path_flair)
            
            affine = nib.load(path_flair).affine
            img = nib.Nifti1Image(seg.astype(np.uint8), affine)
            nib.save(img, f"{save_dir}{id}_{self.team_name}.nii")
            
            if self.mask_exist:
                mask1 = nib.load(path_flair.replace("flair_pp", "mask1").replace("preprocessed", "masks")).get_fdata()
                mask2 = nib.load(path_flair.replace("flair_pp", "mask2").replace("preprocessed", "masks")).get_fdata()
                mask = np.logical_or(mask1, mask2).astype(np.uint8)
                
                row = {"subject": id, "fold": fold, \
                    "dice_scores_mask1": dice_MS_volume(mask1, seg), \
                    "dice_scores_mask2": dice_MS_volume(mask2, seg), \
                    "dice_scores_union": dice_MS_volume(mask, seg)}
                rows_save.append(row)
                
        if self.mask_exist:
            name_csv = f"{save_dir}results_fold_{self.config['TRAIN']['FOLD']}.csv"
            # Open the output CSV file in write mode
            with open(name_csv, 'w', newline='') as csv_file:
                # Write the header row
                fieldnames = ['subject', 'fold', 'dice_scores_mask1', "dice_scores_mask2", "dice_scores_union"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                # Write the rows to the file
                writer.writerows(rows_save)
                
        if self.zip:
            print(f"the results are zipped to: {self.name_archive}.zip")
            shutil.make_archive(self.name_archive, 'zip', self.config["DIRS"]["PREDICT_DIR"])

if __name__ == "__main__":
    data_mean_std = load_yaml("mean_std_isbi2015.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # files to predict
    list_file_flair = sorted(glob.glob("data/data_isbi_2015/training/*/preprocessed/*flair*"))
    # define model
    model = SkipNet(in_dim = config["DATA"]["DIM_SIZE"], num_class=config["DATA"]["NUM_CLASS"])
    
    # create folder to save checkpoints
    os.makedirs(config["DIRS"]["PREDICT_DIR"], exist_ok=True)
    
    # Initialize the segmentation model with the specified parameters
    segmentor = Segmentor(model, config["DATA"]["CLASS_WEIGHT"], config["DATA"]["NUM_CLASS"], 
                                            config["OPT"]["LEARNING_RATE"], config["OPT"]["FACTOR_LR"], config["OPT"]["PATIENCE_LR"])
    
    save_dir = config["DIRS"]["PREDICT_DIR"]
    # Filter out the hidden files
    checkpoint_paths = [config["DIRS"]["SAVE_DIR"]+f for f in sorted(os.listdir(config["DIRS"]["SAVE_DIR"])) if not f.startswith('.')]
    
    # Load checkpoint
    checkpoint = checkpoint_paths[config["PREDICT"]["IDX_CHECKPOINT"]]
    print(f"load checkpoint: {checkpoint}")
    segmentor = Segmentor.load_from_checkpoint(checkpoint_path=checkpoint, model=model,
                                            class_weight=config["DATA"]['CLASS_WEIGHT'],
                                                num_classes=config["DATA"]["NUM_CLASS"], 
                                            learning_rate=config["OPT"]["LEARNING_RATE"],
                                            factor_lr=config["OPT"]["FACTOR_LR"], patience_lr=config["OPT"]["PATIENCE_LR"])
    segmentor = segmentor.to(device)
    segmentor.eval()
    
    # predict
    folder = Predict2Folder(segmentor, list_file_flair, config, data_mean_std, \
                device, team_name="nhatvbd", mask_exist=True, \
                batch_size=config["PREDICT"]["BATCH_SIZE"], make_zip=True, name_archive="testdata1", \
                min_size_remove=config["PREDICT"]["MIN_SIZE_REMOVE"])
    
    folder.submit2folder()