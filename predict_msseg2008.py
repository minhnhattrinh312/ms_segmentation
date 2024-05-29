import nibabel as nib
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
import SimpleITK as sitk


# build a class to predict subjects to a folder
class Predict2Folder:
    def __init__(
        self,
        model,
        list_file_flair,
        checkpoint_list,
        device,
        num_class=2,
        team_name="nhatvbd",
        batch_size=4,
        min_size_remove=100,
        make_zip=True,
        name_archive="testdata1",
    ):
        self.device = device
        self.num_class = num_class
        self.list_file_flair = list_file_flair
        self.team_name = team_name
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

    def predict_subject2d(self, path_flair):
        flair = nib.load(path_flair).get_fdata()
        t1 = nib.load(path_flair.replace("FLAIR", "T1")).get_fdata()
        t2 = nib.load(path_flair.replace("FLAIR", "T2")).get_fdata()

        flair = min_max_normalize(flair)
        t1 = min_max_normalize(t1)
        t2 = min_max_normalize(t2)

        padded_flair, crop_index, padded_index = pad_background(
            flair, dim2pad=cfg.DATA.DIM2PAD_MICCAI2008
        )
        padded_t1 = pad_background_with_index(
            t1, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI2008
        )
        padded_t2 = pad_background_with_index(
            t2, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI2008
        )

        seg = np.zeros((*cfg.DATA.DIM2PAD_MICCAI2008, self.num_class))

        for transpose_view, convert_view in zip(
            cfg.DATA.ORIGIN2CUT, cfg.DATA.CUT2ORIGIN
        ):

            batch_images = []
            output_cur_view = np.zeros((*cfg.DATA.DIM2PAD_MICCAI2008, self.num_class))

            transposed_flair = np.transpose(padded_flair, transpose_view)
            transposed_t1 = np.transpose(padded_t1, transpose_view)
            transposed_t2 = np.transpose(padded_t2, transpose_view)

            for i in range(transposed_flair.shape[-1]):

                slices_flair = transposed_flair[..., i]  # shape (224, 224, 1)
                slices_t1 = transposed_t1[..., i]  # shape (224, 224, 1)
                slices_t2 = transposed_t2[..., i]  # shape (224, 224, 1)
                slice_inputs = np.stack([slices_t1, slices_flair, slices_t2], axis=-1)

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

        seg = np.argmax(seg, axis=-1).astype(np.uint8)
        inverted_image = invert_padding(flair, seg, crop_index, padded_index)
        # post-processing by removing small connected components
        inverted_image = self.remove_small_elements(inverted_image)

        return inverted_image.astype(np.uint8)

    def submit2folder(self):
        save_dir = cfg.DIRS.PREDICT_DIR
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
                    flair = sitk.ReadImage(path_flair)
                    id_subject = path_flair.split("/")[-2]
                    seg = self.predict_subject2d(path_flair)
                    # write sitk.image from seg numpy array
                    sitk_image = sitk.GetImageFromArray(seg)
                    sitk_image.CopyInformation(flair)
                    sitk.WriteImage(
                        sitk_image, f"{save_dir}{id_subject}_segmentation.nrrd"
                    )

                if self.zip:
                    id_weight = checkpoint.split("/")[-1].replace(".ckpt", "")
                    name_archive = self.name_archive + f"_{id_weight}"
                    print(f"the results are zipped to: {name_archive}.zip")
                    shutil.make_archive(
                        name_archive, "zip", "/home/nhattm1/test_newcode", "nhatvinbig"
                    )

        else:

            print("Ensemble prediction")
            for checkpoint in self.checkpoint_list:
                print("Use Checkpoint: ", checkpoint)
            for path_flair in tqdm(self.list_file_flair):
                flair = sitk.ReadImage(path_flair)
                id_subject = path_flair.split("/")[-2]
                seg = self.predict_subject2d(path_flair)
                # write sitk.image from seg numpy array
                sitk_image = sitk.GetImageFromArray(seg)
                sitk_image.CopyInformation(flair)
                sitk.WriteImage(sitk_image, f"{save_dir}{id_subject}_segmentation.nrrd")

            if self.zip:
                print(f"the results are zipped to: {self.name_archive}.zip")
                shutil.make_archive(self.name_archive, "zip", cfg.DIRS.PREDICT_DIR)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # files to predict

    list_file_flair = sorted(glob.glob("data/msseg-2008-testing-nii/*/*FLAIR*"))

    model = FCDenseNet(
        in_channels=cfg.DATA.INDIM_MODEL_MICCAI2008, n_classes=cfg.DATA.NUM_CLASS
    )

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
        checkpoint_list = sorted(
            glob.glob(f"{cfg.DIRS.SAVE_DIR}fold{cfg.TRAIN.FOLD}/*.ckpt")
        )
        # print(checkpoint_list)

    # Load checkpoint
    # predict
    folder = Predict2Folder(
        model,
        list_file_flair,
        checkpoint_list,
        device,
        team_name="nhatvinbig",
        batch_size=cfg.PREDICT.BATCH_SIZE,
        make_zip=True,
        name_archive=name_zip,
        min_size_remove=cfg.PREDICT.MIN_SIZE_REMOVE,
    )

    folder.submit2folder()
