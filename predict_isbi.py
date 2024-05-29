import nibabel as nib

data_mean_std = None
# from segment3d import *
from segment2d import *
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
import segmentation_models_pytorch as smp


# build a class to predict subjects to a folder
class Predict2Folder:
    def __init__(
        self,
        model,
        list_file_flair,
        data_mean_std,
        checkpoint_list,
        device,
        num_class=2,
        team_name="nhatvbd",
        mask_exist=True,
        batch_size=4,
        min_size_remove=100,
        make_zip=True,
        name_archive="testdata1",
    ):
        self.data_mean_std = data_mean_std
        self.device = device
        self.num_class = num_class
        self.list_file_flair = list_file_flair
        self.team_name = team_name
        self.mask_exist = mask_exist
        self.batch_size = batch_size
        self.min_size_remove = min_size_remove
        self.name_archive = name_archive
        self.zip = make_zip
        self.checkpoint_list = checkpoint_list
        self.model = model
        self.segmenter = Segmenter(
            model,
            cfg.DATA.CLASS_WEIGHT,
            cfg.DATA.NUM_CLASS,
            cfg.OPT.LEARNING_RATE,
            cfg.OPT.FACTOR_LR,
            cfg.OPT.PATIENCE_LR,
        )

    def predict_patches(self, images):
        """return the patches"""
        # images = torch.from_numpy(images)
        if cfg.PREDICT.MODE == "3D":
            prediction = torch.zeros(
                (
                    images.size(0),
                    self.num_class,
                    images.size(2),
                    images.size(3),
                    images.size(4),
                ),
                device=self.device,
            )
        else:
            prediction = torch.zeros(
                (images.size(0), self.num_class, images.size(2), images.size(3)),
                device=self.device,
            )

        batch_start = 0
        batch_end = self.batch_size
        while batch_start < images.size(0):
            image = images[batch_start:batch_end]
            with torch.inference_mode():
                image = image.to(self.device)
                y_pred = self.segmenter(image)
                prediction[batch_start:batch_end] = y_pred
            batch_start += self.batch_size
            batch_end += self.batch_size

        return prediction.cpu().numpy()

    def remove_small_elements(self, segmentation_mask):
        # Convert segmentation mask values greater than 0 to 1
        pred_mask = segmentation_mask > 0
        # Remove small objects (connected components) from the binary image
        mask = remove_small_objects(pred_mask, min_size=self.min_size_remove)
        # Multiply original segmentation mask with the mask to remove small objects
        clean_segmentation = segmentation_mask * mask
        return clean_segmentation
        # seg_labels, seg_num = measure.label(segmentation_mask, return_num=True, connectivity=2)

        # for label in range(1, seg_num + 1):
        #     tmp_cnt = np.sum(segmentation_mask[seg_labels == label])
        #     # print(tmp_cnt)
        #     tmp_cnt = int(tmp_cnt)
        #     if tmp_cnt < self.min_size_remove:
        #         segmentation_mask[seg_labels == label] = 0

        # return segmentation_mask

    def predict_subject2d(self, path_flair):
        flair = nib.load(path_flair).get_fdata()
        t1 = nib.load(path_flair.replace("flair", "mprage")).get_fdata()
        pd = nib.load(path_flair.replace("flair", "pd")).get_fdata()
        t2 = nib.load(path_flair.replace("flair", "t2")).get_fdata()

        flair = min_max_normalize(flair)
        t1 = min_max_normalize(t1)
        pd = min_max_normalize(pd)
        t2 = min_max_normalize(t2)
        id = path_flair.split("/")[-1]
        parts = id.split("_")
        # Access the first two elements in the list (index 0 and 1)
        id = parts[0] + "_" + parts[1]

        padded_flair, crop_index, padded_index = pad_background(
            flair, dim2pad=cfg.DATA.DIM2PAD_ISBI
        )
        padded_t1 = pad_background_with_index(
            t1, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI
        )
        padded_pd = pad_background_with_index(
            pd, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI
        )
        padded_t2 = pad_background_with_index(
            t2, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI
        )

        seg = np.zeros((*cfg.DATA.DIM2PAD_ISBI, self.num_class))
        # seg = np.zeros(cfg.DATA.DIM2PAD_ISBI)

        view = 0

        for transpose_view, convert_view in zip(
            cfg.DATA.ORIGIN2CUT, cfg.DATA.CUT2ORIGIN
        ):

            batch_images = []
            output_cur_view = np.zeros((*cfg.DATA.DIM2PAD_ISBI, self.num_class))
            # output_cur_view = np.zeros(cfg.DATA.DIM2PAD_ISBI)
            transposed_flair = np.transpose(padded_flair, transpose_view)
            transposed_t1 = np.transpose(padded_t1, transpose_view)
            transposed_t2 = np.transpose(padded_t2, transpose_view)
            transposed_pd = np.transpose(padded_pd, transpose_view)

            for i in range(transposed_flair.shape[-1]):

                slices_flair = transposed_flair[..., i]  # shape (224, 224, 1)
                slices_t1 = transposed_t1[..., i]  # shape (224, 224, 1)
                slices_t2 = transposed_t2[..., i]  # shape (224, 224, 1)
                slices_pd = transposed_pd[..., i]  # shape (224, 224, 1)
                slice_inputs = np.stack(
                    [slices_t1, slices_flair, slices_t2, slices_pd], axis=-1
                )

                slices_image = torch.from_numpy(
                    slice_inputs.transpose(-1, 0, 1)
                )  # shape (3, 224, 224

                batch_images.append(slices_image)

            batch_images = torch.stack(batch_images).float()  # shape (n, 12, 224, 224)
            probability_output = []

            if cfg.PREDICT.ENSEMBLE:
                for checkpoint, weight_ckpt in zip(
                    self.checkpoint_list, cfg.PREDICT.WEIGHTS
                ):
                    self.segmenter = Segmenter(
                        self.model,
                        cfg.DATA.CLASS_WEIGHT,
                        cfg.DATA.NUM_CLASS,
                        cfg.OPT.LEARNING_RATE,
                        cfg.OPT.FACTOR_LR,
                        cfg.OPT.PATIENCE_LR,
                    )
                    self.segmenter = Segmenter.load_from_checkpoint(
                        checkpoint_path=checkpoint,
                        model=self.model,
                        class_weight=cfg.DATA["CLASS_WEIGHT"],
                        num_classes=cfg.DATA.NUM_CLASS,
                        learning_rate=cfg.OPT.LEARNING_RATE,
                        factor_lr=cfg.OPT.FACTOR_LR,
                        patience_lr=cfg.OPT.PATIENCE_LR,
                    )
                    self.segmenter = self.segmenter.to(self.device)
                    self.segmenter.eval()
                    y_pred = self.predict_patches(
                        batch_images
                    )  # shape (n, 2, 224, 224)
                    probability_output.append(y_pred * weight_ckpt)

                probability_output = np.stack(
                    probability_output, axis=0
                )  # shape (num_fold, n, 2, 224, 224)
                probability_output = np.sum(
                    probability_output, axis=0
                )  # shape (n, 2, 224, 224)

            else:
                probability_output = self.predict_patches(
                    batch_images
                )  # shape (n, 2, 224, 224)
            probability_output = probability_output.transpose(
                2, 3, 0, 1
            )  # shape (224, 224, n, 2)
            output_cur_view = (
                probability_output  # output_cur_view has shape (224, 224, 224, 2)
            )
            convert_view = convert_view + (-1,)

            output_cur_view_trans = np.transpose(
                output_cur_view, convert_view
            )  # convert to original view
            seg += output_cur_view_trans

            if self.mask_exist:
                view += 1
                save_dir = cfg.DIRS.PREDICT_DIR
                np.save(f"{save_dir}{id}_view{view}.npy", output_cur_view[padded_index])

        ###########################################

        seg = np.argmax(seg, axis=-1).astype(np.uint8)
        inverted_image = invert_padding(flair, seg, crop_index, padded_index)
        # post-processing by removing small connected components
        inverted_image = self.remove_small_elements(inverted_image)

        return inverted_image

    def submit2folder(self):
        save_dir = cfg.DIRS.PREDICT_DIR
        rows_save = []
        if not cfg.PREDICT.ENSEMBLE:
            print("Single prediction")
            for checkpoint in self.checkpoint_list:
                print("Use Checkpoint: ", checkpoint)
                self.segmenter = Segmenter.load_from_checkpoint(
                    checkpoint_path=checkpoint,
                    model=self.model,
                    class_weight=cfg.DATA.CLASS_WEIGHT,
                    num_classes=cfg.DATA.NUM_CLASS,
                    learning_rate=cfg.OPT.LEARNING_RATE,
                    factor_lr=cfg.OPT.FACTOR_LR,
                    patience_lr=cfg.OPT.PATIENCE_LR,
                )
                self.segmenter = self.segmenter.to(self.device)
                self.segmenter.eval()

                for path_flair in tqdm(self.list_file_flair):
                    id = path_flair.split("/")[-1]
                    parts = id.split("_")
                    # Access the first two elements in the list (index 0 and 1)
                    id = parts[0] + "_" + parts[1]
                    fold = int(parts[0][-2:])
                    if cfg.PREDICT.MODE == "3D":
                        seg = self.predict_subject3d(path_flair)
                    else:
                        seg = self.predict_subject2d(path_flair)

                    affine = nib.load(path_flair).affine
                    img = nib.Nifti1Image(seg.astype(np.uint8), affine)
                    nib.save(img, f"{save_dir}{id}_{self.team_name}.nii")

                    if self.mask_exist:
                        mask1 = nib.load(
                            path_flair.replace("flair_pp", "mask1").replace(
                                "preprocessed", "masks"
                            )
                        ).get_fdata()
                        mask2 = nib.load(
                            path_flair.replace("flair_pp", "mask2").replace(
                                "preprocessed", "masks"
                            )
                        ).get_fdata()
                        mask = np.logical_or(mask1, mask2).astype(np.int64)

                        row = {
                            "subject": id,
                            "fold": fold,
                            "dice_scores_mask1": dice_MS_volume(mask1, seg),
                            "dice_scores_mask2": dice_MS_volume(mask2, seg),
                            "dice_scores_union": dice_MS_volume(mask, seg),
                        }
                        rows_save.append(row)
                if self.zip:
                    id_weight = checkpoint.split("/")[-1].replace(".ckpt", "")
                    name_archive = self.name_archive + f"_{id_weight}"
                    print(f"the results are zipped to: {name_archive}.zip")
                    shutil.make_archive(name_archive, "zip", cfg.DIRS.PREDICT_DIR)

        else:

            print("Ensemble prediction")
            for checkpoint in self.checkpoint_list:
                print("Use Checkpoint: ", checkpoint)
            for path_flair in tqdm(self.list_file_flair):
                id = path_flair.split("/")[-1]
                parts = id.split("_")
                # Access the first two elements in the list (index 0 and 1)
                id = parts[0] + "_" + parts[1]

                if cfg.PREDICT.MODE == "3D":
                    seg = self.predict_subject3d(path_flair)
                else:
                    seg = self.predict_subject2d(path_flair)

                affine = nib.load(path_flair).affine
                img = nib.Nifti1Image(seg.astype(np.uint8), affine)
                nib.save(img, f"{save_dir}{id}_{self.team_name}.nii")

            if self.zip:
                print(f"the results are zipped to: {self.name_archive}.zip")
                shutil.make_archive(self.name_archive, "zip", cfg.DIRS.PREDICT_DIR)

        if self.mask_exist:
            name_csv = f"{save_dir}results_fold_{cfg.TRAIN.FOLD}.csv"
            # Open the output CSV file in write mode
            with open(name_csv, "w", newline="") as csv_file:
                # Write the header row
                fieldnames = [
                    "subject",
                    "fold",
                    "dice_scores_mask1",
                    "dice_scores_mask2",
                    "dice_scores_union",
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                # Write the rows to the file
                writer.writerows(rows_save)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # files to predict
    if not cfg.PREDICT.MASK_EXIST:
        list_file_flair = sorted(
            glob.glob("data/data_isbi_2015/testdata_website/*/preprocessed/*flair*")
        )
    else:
        list_file_flair = sorted(
            glob.glob("data/data_isbi_2015/training/*/preprocessed/*flair*")
        )
    # define model
    if cfg.PREDICT.MODEL == "convnext":
        print("Use ConvNext")
        model = SkipNet(in_dim=cfg.DATA.DIM_SIZE, num_class=cfg.DATA.NUM_CLASS)
    elif cfg.PREDICT.MODEL == "resnet50":
        print("Use ResNet50")
        model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=cfg.DATA.INDIM_MODEL,
            classes=2,
            activation="softmax2d",
        )
    elif cfg.PREDICT.MODEL == "tiramisu":
        model = FCDenseNet(
            in_channels=cfg.DATA.INDIM_MODEL, n_classes=cfg.DATA.NUM_CLASS
        )
    else:
        print("no model is loaded!!!!!!!!!!!")
    # create folder to save checkpoints
    os.makedirs(cfg.DIRS.PREDICT_DIR, exist_ok=True)

    # Initialize the segmentation model with the specified parameters

    path_folds = sorted(glob.glob(cfg.DIRS.SAVE_DIR[:-2] + "*/"))
    if cfg.PREDICT.ENSEMBLE:
        name_zip = cfg.PREDICT.NAME_ZIP
        checkpoint_list = [
            sorted(glob.glob(path_fold + "*.ckpt"))[-1] for path_fold in path_folds
        ]
        # remove fold 2 from the list
        # checkpoint_list = [checkpoint for checkpoint in checkpoint_list if "fold3" not in checkpoint]
        # checkpoint_list = [checkpoint for checkpoint in checkpoint_list if "fold3" not in checkpoint]
        # checkpoint_list = [checkpoint for checkpoint in checkpoint_list if "fold1" not in checkpoint]
    else:
        name_zip = cfg.PREDICT.MODEL + str(cfg.TRAIN.FOLD)
        checkpoint_list = sorted(glob.glob(cfg.DIRS.SAVE_DIR + "*.ckpt"))

    # Load checkpoint
    # predict
    folder = Predict2Folder(
        model,
        list_file_flair,
        data_mean_std,
        checkpoint_list,
        device,
        team_name="nhatvin",
        mask_exist=cfg.PREDICT.MASK_EXIST,
        batch_size=cfg.PREDICT.BATCH_SIZE,
        make_zip=True,
        name_archive=name_zip,
        min_size_remove=cfg.PREDICT.MIN_SIZE_REMOVE,
    )

    folder.submit2folder()
