import glob
from torch.utils.data import Dataset
import torch
import numpy as np
from segment2d.utils import *
from segment2d.config import cfg
import nibabel as nib


class ISBILoader(Dataset):

    def __init__(self, train_path="", list_subject=[]):
        if list_subject:
            self.listName = list_subject
        else:
            self.listName = glob.glob(train_path)

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        data = np.load(self.listName[idx])
        image, mask = data["flair"], data["mask"]
        # convert image, mask to tensor
        image = torch.from_numpy(image.transpose(-1, 0, 1))

        mask = torch.from_numpy(mask[None])
        return image, mask.float()


class MSSEGLoader(Dataset):

    def __init__(self, train_path="", list_subject=[]):
        if list_subject:
            self.listName = list_subject
        else:
            self.listName = glob.glob(train_path)

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        data = np.load(self.listName[idx])
        image, mask = data["flair"], data["mask"]
        # convert image, mask to tensor
        image = torch.from_numpy(image.transpose(-1, 0, 1))

        mask = torch.from_numpy(mask[None])
        return image, mask.float()


class ISBI_Test_Loader(Dataset):

    def __init__(self, list_subject=[]):

        self.listName = list_subject

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        path_flair = self.listName[idx]
        data = dict()

        flair = nib.load(path_flair).get_fdata()
        t1 = nib.load(path_flair.replace("flair", "mprage")).get_fdata()
        pd = nib.load(path_flair.replace("flair", "pd")).get_fdata()
        t2 = nib.load(path_flair.replace("flair", "t2")).get_fdata()

        flair = min_max_normalize(flair)
        t1 = min_max_normalize(t1)
        pd = min_max_normalize(pd)
        t2 = min_max_normalize(t2)

        mask1 = nib.load(
            path_flair.replace("flair_pp", "mask1").replace("preprocessed", "masks")
        )
        mask2 = nib.load(
            path_flair.replace("flair_pp", "mask2").replace("preprocessed", "masks")
        )

        mask1 = mask1.get_fdata().astype(np.uint8)
        mask2 = mask2.get_fdata().astype(np.uint8)

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
        padded_mask1 = pad_background_with_index(
            mask1, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI
        )
        padded_mask2 = pad_background_with_index(
            mask2, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_ISBI
        )

        data["mask1"] = padded_mask1.astype(np.int64)
        data["mask2"] = padded_mask2.astype(np.int64)

        views = ["coronal", "sagittal", "axial"]
        for transpose_view, view in zip(cfg.DATA.ORIGIN2CUT, views):
            transposed_flair = np.transpose(padded_flair, transpose_view)
            transposed_t1 = np.transpose(padded_t1, transpose_view)
            transposed_t2 = np.transpose(padded_t2, transpose_view)
            transposed_pd = np.transpose(padded_pd, transpose_view)

            batch_images = []

            for i in range(transposed_flair.shape[-1]):
                slices_flair = transposed_flair[..., i]  # shape (224, 224, 3)
                slices_t1 = transposed_t1[..., i]  # shape (224, 224, 3)
                slices_t2 = transposed_t2[..., i]  # shape (224, 224, 3)
                slices_pd = transposed_pd[..., i]  # shape (224, 224, 3)

                slice_inputs = np.stack(
                    [slices_t1, slices_flair, slices_t2, slices_pd], axis=-1
                )  # shape (224, 224, 4)
                slices_image = torch.from_numpy(
                    slice_inputs.transpose(-1, 0, 1)
                )  # shape (3, 224, 224)

                batch_images.append(slices_image)

            batch_images = torch.stack(batch_images).float()  # shape (224, 4, 224, 224)
            data[view] = batch_images

        return data


class MSSEG_Test_Loader(Dataset):

    def __init__(self, list_subject=[]):

        self.listName = list_subject

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        path_flair = self.listName[idx]
        data = dict()

        flair = nib.load(path_flair).get_fdata()
        t1 = nib.load(path_flair.replace("FLAIR", "T1")).get_fdata()
        t2 = nib.load(path_flair.replace("FLAIR", "T2")).get_fdata()
        dp = nib.load(path_flair.replace("FLAIR", "DP")).get_fdata()
        gado = nib.load(path_flair.replace("FLAIR", "GADO")).get_fdata()

        flair = min_max_normalize(flair)
        t1 = min_max_normalize(t1)
        t2 = min_max_normalize(t2)
        dp = min_max_normalize(dp)
        gado = min_max_normalize(gado)

        consensus = nib.load(
            path_flair.replace("Preprocessed_Data", "Masks").replace(
                "FLAIR_preprocessed", "Consensus"
            )
        )
        consensus = consensus.get_fdata().astype(np.uint64)

        padded_flair, crop_index, padded_index = pad_background(
            flair, dim2pad=cfg.DATA.DIM2PAD_MICCAI
        )
        padded_t1 = pad_background_with_index(
            t1, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI
        )
        padded_dp = pad_background_with_index(
            dp, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI
        )
        padded_t2 = pad_background_with_index(
            t2, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI
        )
        padded_gado = pad_background_with_index(
            gado, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI
        )
        # padded_consensus = pad_background_with_index(consensus, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI)

        # data["padded_consensus"] = padded_consensus.astype(np.int64)
        data["consensus"] = consensus.astype(np.int64)
        data["crop_index"] = crop_index
        data["padded_index"] = padded_index

        views = ["coronal", "sagittal", "axial"]
        for view, transpose_view in zip(views, cfg.DATA.ORIGIN2CUT):

            batch_images = []
            transposed_flair = np.transpose(padded_flair, transpose_view)
            transposed_t1 = np.transpose(padded_t1, transpose_view)
            transposed_t2 = np.transpose(padded_t2, transpose_view)
            transposed_dp = np.transpose(padded_dp, transpose_view)
            transposed_gado = np.transpose(padded_gado, transpose_view)

            for i in range(transposed_flair.shape[-1]):
                slices_flair = transposed_flair[..., i]  # shape (256, 256, 1)
                slices_t1 = transposed_t1[..., i]  # shape (224, 224, 1)
                slices_t2 = transposed_t2[..., i]  # shape (224, 224, 1)
                slices_dp = transposed_dp[..., i]  # shape (224, 224, 1)
                slices_gado = transposed_gado[..., i]  # shape (224, 224, 1)
                slice_inputs = np.stack(
                    [slices_t1, slices_flair, slices_t2, slices_dp, slices_gado],
                    axis=-1,
                )  # shape (224, 224, 4)

                slices_image = torch.from_numpy(
                    slice_inputs.transpose(-1, 0, 1)
                )  # shape (5, 256, 256)

                batch_images.append(slices_image)

            batch_images = torch.stack(batch_images).float()  # shape (256, 5, 256, 256)
            data[view] = batch_images

        return data


class MSSEG2008_Test_Loader(Dataset):

    def __init__(self, list_subject=[]):

        self.listName = list_subject

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        path_flair = self.listName[idx]
        data = dict()

        flair = nib.load(path_flair).get_fdata()
        t1 = nib.load(path_flair.replace("FLAIR", "T1")).get_fdata()
        t2 = nib.load(path_flair.replace("FLAIR", "T2")).get_fdata()

        flair = min_max_normalize(flair)
        t1 = min_max_normalize(t1)
        t2 = min_max_normalize(t2)

        mask1 = nib.load(path_flair.replace("FLAIR_brain_bias_correction", "lesion"))
        mask1 = mask1.get_fdata().astype(np.uint8)

        padded_flair, crop_index, padded_index = pad_background(
            flair, dim2pad=cfg.DATA.DIM2PAD_MICCAI2008
        )
        padded_t1 = pad_background_with_index(
            t1, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI2008
        )
        padded_t2 = pad_background_with_index(
            t2, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI2008
        )
        padded_mask1 = pad_background_with_index(
            mask1, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI2008
        )
        if "UNC" in path_flair:
            mask2 = nib.load(
                path_flair.replace("FLAIR_brain_bias_correction", "lesion_byCHB")
            )
            mask2 = mask2.get_fdata().astype(np.uint8)
            padded_mask2 = pad_background_with_index(
                mask2, crop_index, padded_index, dim2pad=cfg.DATA.DIM2PAD_MICCAI2008
            )
            data["mask2"] = padded_mask2.astype(np.int64)

        data["mask1"] = padded_mask1.astype(np.int64)

        views = ["coronal", "sagittal", "axial"]
        for view, transpose_view in zip(views, cfg.DATA.ORIGIN2CUT):

            batch_images = []
            transposed_flair = np.transpose(padded_flair, transpose_view)
            transposed_t1 = np.transpose(padded_t1, transpose_view)
            transposed_t2 = np.transpose(padded_t2, transpose_view)

            for i in range(transposed_flair.shape[-1]):
                slices_flair = transposed_flair[..., i]  # shape (256, 256, 1)
                slices_t1 = transposed_t1[..., i]  # shape (224, 224, 1)
                slices_t2 = transposed_t2[..., i]  # shape (224, 224, 1)
                slice_inputs = np.stack(
                    [slices_t1, slices_flair, slices_t2], axis=-1
                )  # shape (224, 224, 4)

                slices_image = torch.from_numpy(
                    slice_inputs.transpose(-1, 0, 1)
                )  # shape (5, 256, 256)

                batch_images.append(slices_image)

            batch_images = torch.stack(batch_images).float()  # shape (256, 5, 256, 256)
            data[view] = batch_images

        return data
