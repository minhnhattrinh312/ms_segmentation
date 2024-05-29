import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm.optim
import pytorch_lightning as pl
from segment2d.utils import *
from segment2d.losses import *
from segment2d.metrics import *
from segment2d.config import cfg
import torch.nn.functional as F
from kornia.augmentation import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomAffine,
    AugmentationSequential,
    Normalize,
)
import kornia as K
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau


class Segmenter(pl.LightningModule):
    def __init__(
        self,
        model,
        class_weight,
        num_classes,
        learning_rate,
        factor_lr,
        patience_lr,
        batch_size_predict=16,
    ):
        super().__init__()
        self.model = model
        # torch 2.3 => compile to make faster
        self.model  = torch.compile(self.model, mode="reduce-overhead")
        
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.factor_lr = factor_lr
        self.patience_lr = patience_lr
        self.batch_size = batch_size_predict
        ################ augmentation ############################
        self.transform = AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomAffine(degrees=10, translate=0.0625, scale=(0.95, 1.05), p=0.5),
            data_keys=["input", "mask"],
        )
        self.test_metric = []
        self.validation_step_outputs = []

    def on_train_start(self):
        if cfg.TRAIN.LOSS == "active_focal":
            self.training_loss = ActiveFocalLoss(
                self.device, self.class_weight, self.num_classes
            )
        elif cfg.TRAIN.LOSS == "active_contour":
            self.training_loss = ActiveContourLoss(
                self.device, self.class_weight, self.num_classes
            )
        elif cfg.TRAIN.LOSS == "CrossEntropy":
            self.training_loss = CrossEntropy(self.device, self.num_classes)
        elif cfg.TRAIN.LOSS == "DiceLoss":
            self.training_loss = DiceLoss(self.device, self.num_classes)
        elif cfg.TRAIN.LOSS == "TverskyLoss":
            self.training_loss = TverskyLoss(self.device, self.num_classes)
        elif cfg.TRAIN.LOSS == "MSELoss":
            self.training_loss = MSELoss(self.device, self.num_classes)
        else:
            self.training_loss = ActiveFocalContourLoss(
                self.device, self.class_weight, self.num_classes
            )

    def forward(self, x):
        # return self.model(self.normalize(x))
        return self.model(x)

    def predict_patches(self, images):
        """return the patches"""
        prediction = torch.zeros(
            (images.size(0), self.num_classes, images.size(2), images.size(3)),
            device=self.device,
        )

        batch_start = 0
        batch_end = self.batch_size
        while batch_start < images.size(0):
            image = images[batch_start:batch_end]
            with torch.inference_mode():
                image = image.to(self.device)
                y_pred = self.model(image)
                prediction[batch_start:batch_end] = y_pred
            batch_start += self.batch_size
            batch_end += self.batch_size
        return prediction.cpu().numpy()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            with torch.no_grad():
                batch = self.transform(*batch)
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.trainer.training:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def _step(self, batch):
        image, y_true = batch

        y_pred = self.model(image)
        loss = self.training_loss(y_true, y_pred)
        dice_ms = dice_MS(y_true, y_pred)
        return loss, dice_ms

    def training_step(self, batch, batch_idx):
        loss, dice_ms = self._step(batch)
        metrics = {"losses": loss, "diceTrain": dice_ms}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step_isbi(self, batch, batch_idx):
        # batch is a dictionary of tensors ["coronal", "sagittal", "axial", "mask1", "mask2"]
        views = ["coronal", "sagittal", "axial"]
        seg = np.zeros((*cfg.DATA.DIM2PAD_ISBI, self.num_classes))

        for view, convert_view in zip(views, cfg.DATA.CUT2ORIGIN):
            output_cur_view = np.zeros((*cfg.DATA.DIM2PAD_ISBI, self.num_classes))

            probability_output = self.predict_patches(
                batch[view]
            )  # shape (n, 2, 224, 224)
            probability_output = probability_output.transpose(
                2, 3, 0, 1
            )  # shape (224, 224, n, 2)
            output_cur_view = probability_output
            convert_view = convert_view + (-1,)
            output_cur_view_trans = np.transpose(
                output_cur_view, convert_view
            )  # convert to original view
            seg += output_cur_view_trans

        seg = np.argmax(seg, axis=-1).astype(np.uint8)
        metrics_mask1 = seg_metrics(batch["mask1"], seg)
        metrics_mask2 = seg_metrics(batch["mask2"], seg)
        metrics = {
            "batch_score_val": np.sum(
                [metrics_mask1["isbi_score"], metrics_mask2["isbi_score"]]
            ),
            "batch_dice_val": np.sum([metrics_mask1["dice"], metrics_mask2["dice"]]),
        }
        return metrics

    def validation_step_msseg2016(self, batch, batch_idx):
        # batch is a dictionary of tensors ["coronal", "sagittal", "axial", "mask1", "mask2"]
        views = ["coronal", "sagittal", "axial"]

        seg = np.zeros((*cfg.DATA.DIM2PAD_MICCAI, self.num_classes))

        for view, convert_view in zip(views, cfg.DATA.CUT2ORIGIN):
            output_cur_view = np.zeros((*cfg.DATA.DIM2PAD_MICCAI, self.num_classes))
            probability_output = self.predict_patches(
                batch[view]
            )  # shape (n, 2, 224, 224)
            probability_output = probability_output.transpose(
                2, 3, 0, 1
            )  # shape (224, 224, n, 2)
            output_cur_view = probability_output
            convert_view = convert_view + (-1,)
            output_cur_view_trans = np.transpose(
                output_cur_view, convert_view
            )  # convert to original view
            seg += output_cur_view_trans

        seg = np.argmax(seg, axis=-1).astype(np.uint8)
        seg = remove_small_elements(seg, min_size_remove=cfg.PREDICT.MIN_SIZE_REMOVE)
        inverted_image = invert_padding(
            batch["consensus"], seg, batch["crop_index"], batch["padded_index"]
        )
        metrics_mask = seg_metrics(batch["consensus"], inverted_image)
        metrics = {
            "batch_score_val": metrics_mask["isbi_score"],
            "batch_dice_val": metrics_mask["dice"],
            "batch_ppv_val": metrics_mask["ppv"],
            "batch_tpr_val": metrics_mask["tpr"],
            "batch_lfpr_val": metrics_mask["lfpr"],
            "batch_ltpr_val": metrics_mask["ltpr"],
            "batch_vd_val": metrics_mask["vd"],
        }
        return metrics

    def validation_step_msseg2008(self, batch, batch_idx):
        # batch is a dictionary of tensors ["coronal", "sagittal", "axial", "mask1", "mask2"]
        views = ["coronal", "sagittal", "axial"]

        seg = np.zeros((*cfg.DATA.DIM2PAD_MICCAI2008, self.num_classes))

        for view, convert_view in zip(views, cfg.DATA.CUT2ORIGIN):
            output_cur_view = np.zeros((*cfg.DATA.DIM2PAD_MICCAI2008, self.num_classes))
            probability_output = self.predict_patches(
                batch[view]
            )  # shape (n, 2, 224, 224)
            probability_output = probability_output.transpose(
                2, 3, 0, 1
            )  # shape (224, 224, n, 2)
            output_cur_view = probability_output
            convert_view = convert_view + (-1,)
            output_cur_view_trans = np.transpose(
                output_cur_view, convert_view
            )  # convert to original view
            seg += output_cur_view_trans

        seg = np.argmax(seg, axis=-1).astype(np.uint8)
        # seg = remove_small_elements(seg, min_size_remove=cfg.PREDICT.MIN_SIZE_REMOVE)
        metrics_mask1 = seg_metrics_miccai2008(batch["mask1"], seg)
        metrics = {
            "batch_score_val": metrics_mask1["score"],
            "batch_tpr_val": metrics_mask1["tpr"],
            "batch_fpr_val": metrics_mask1["fpr"],
            "batch_vd_val": metrics_mask1["vd"],
        }

        if "mask2" in batch:
            metrics_mask2 = seg_metrics_miccai2008(batch["mask1"], seg)
            metrics = {
                "batch_score_val": np.mean(
                    [metrics_mask1["score"], metrics_mask2["score"]]
                ),
                "batch_tpr_val": np.mean([metrics_mask1["tpr"], metrics_mask2["tpr"]]),
                "batch_fpr_val": np.mean([metrics_mask1["fpr"], metrics_mask2["fpr"]]),
                "batch_vd_val": np.mean([metrics_mask1["vd"], metrics_mask2["vd"]]),
            }
        return metrics

    def validation_step(self, batch, batch_idx):
        if cfg.TRAIN.TASK == "isbi":
            pred = self.validation_step_isbi(batch, batch_idx)
            self.validation_step_outputs.append(pred)
            return pred

        elif cfg.TRAIN.TASK == "msseg":
            pred = self.validation_step_msseg2016(batch, batch_idx)
            self.validation_step_outputs.append(pred)
            return pred
        else:
            pred = self.validation_step_msseg2008(batch, batch_idx)
            self.validation_step_outputs.append(pred)
            return pred

    def on_validation_epoch_end(self):
        if cfg.TRAIN.TASK == "isbi":
            avg_score = (
                np.stack(
                    [x["batch_score_val"] for x in self.validation_step_outputs]
                ).mean()
                / 2
            )
            avg_dice = (
                np.stack(
                    [x["batch_dice_val"] for x in self.validation_step_outputs]
                ).mean()
                / 2
            )
            metrics = {"val_score": avg_score, "val_dice": avg_dice}
        elif cfg.TRAIN.TASK == "msseg":
            avg_score = np.stack(
                [x["batch_score_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_dice = np.stack(
                [x["batch_dice_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_ppv = np.stack(
                [x["batch_ppv_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_tpr = np.stack(
                [x["batch_tpr_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_lfpr = np.stack(
                [x["batch_lfpr_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_ltpr = np.stack(
                [x["batch_ltpr_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_vd = np.stack(
                [x["batch_vd_val"] for x in self.validation_step_outputs]
            ).mean()
            metrics = {
                "test_score": avg_score,
                "test_dice": avg_dice,
                "test_ppv": avg_ppv,
                "test_tpr": avg_tpr,
                "test_lfpr": avg_lfpr,
                "test_ltpr": avg_ltpr,
                "test_vd": avg_vd,
            }
        else:  # for mseeg2008
            avg_score = np.stack(
                [x["batch_score_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_tpr = np.stack(
                [x["batch_tpr_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_fpr = np.stack(
                [x["batch_fpr_val"] for x in self.validation_step_outputs]
            ).mean()
            avg_vd = np.stack(
                [x["batch_vd_val"] for x in self.validation_step_outputs]
            ).mean()
            metrics = {
                "val_score": avg_score,
                "val_tpr": avg_tpr,
                "val_fpr": avg_fpr,
                "val_vd": avg_vd,
            }
        self.log_dict(metrics, prog_bar=True)

        return metrics

    def test_step(self, batch, batch_idx):  # msseg2016 only
        if cfg.TRAIN.TASK == "msseg":
            # batch is a dictionary of tensors ["coronal", "sagittal", "axial", "mask1", "mask2"]
            views = ["coronal", "sagittal", "axial"]
            seg = np.zeros((*cfg.DATA.DIM2PAD_MICCAI, self.num_classes))

            for view, convert_view in zip(views, cfg.DATA.CUT2ORIGIN):
                output_cur_view = np.zeros((*cfg.DATA.DIM2PAD_MICCAI, self.num_classes))

                probability_output = self.predict_patches(
                    batch[view]
                )  # shape (n, 2, 224, 224)
                probability_output = probability_output.transpose(
                    2, 3, 0, 1
                )  # shape (224, 224, n, 2)
                output_cur_view = probability_output
                convert_view = convert_view + (-1,)
                output_cur_view_trans = np.transpose(
                    output_cur_view, convert_view
                )  # convert to original view
                seg += output_cur_view_trans
                # break

            seg = np.argmax(seg, axis=-1).astype(np.uint8)
            seg = remove_small_elements(
                seg, min_size_remove=cfg.PREDICT.MIN_SIZE_REMOVE
            )
            # print(np.sum(seg))
            inverted_image = invert_padding(
                batch["consensus"], seg, batch["crop_index"], batch["padded_index"]
            )
            metrics_mask = seg_metrics(batch["consensus"], inverted_image)
            metrics = {
                "batch_score_val": metrics_mask["isbi_score"],
                "batch_dice_val": metrics_mask["dice"],
                "batch_ppv_val": metrics_mask["ppv"],
                "batch_tpr_val": metrics_mask["tpr"],
                "batch_lfpr_val": metrics_mask["lfpr"],
                "batch_ltpr_val": metrics_mask["ltpr"],
                "batch_vd_val": metrics_mask["vd"],
            }
            print(metrics)
            self.test_metric.append(metrics)
            return metrics

    def on_test_end(self):
        avg_score = np.stack([x["batch_score_val"] for x in self.test_metric]).mean()
        avg_dice = np.stack([x["batch_dice_val"] for x in self.test_metric]).mean()
        avg_ppv = np.stack([x["batch_ppv_val"] for x in self.test_metric]).mean()
        avg_tpr = np.stack([x["batch_tpr_val"] for x in self.test_metric]).mean()
        avg_lfpr = np.stack([x["batch_lfpr_val"] for x in self.test_metric]).mean()
        avg_ltpr = np.stack([x["batch_ltpr_val"] for x in self.test_metric]).mean()
        avg_vd = np.stack([x["batch_vd_val"] for x in self.test_metric]).mean()
        metrics = {
            "test_score": avg_score,
            "test_dice": avg_dice,
            "test_ppv": avg_ppv,
            "test_tpr": avg_tpr,
            "test_lfpr": avg_lfpr,
            "test_ltpr": avg_ltpr,
            "test_vd": avg_vd,
        }
        print(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = timm.optim.Nadam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=self.factor_lr, patience=self.patience_lr
        )
        if cfg.TRAIN.TASK == "isbi":
            lr_schedulers = {
                "scheduler": scheduler,
                "monitor": "val_score",
                "strict": False,
            }
        elif cfg.TRAIN.TASK == "msseg":
            lr_schedulers = {
                "scheduler": scheduler,
                "monitor": "test_dice",
                "strict": False,
            }
        else:  # for mseeg2008
            lr_schedulers = {
                "scheduler": scheduler,
                "monitor": "val_score",
                "strict": False,
            }
        # return [optimizer]
        return [optimizer], lr_schedulers

    def lr_scheduler_step(self, scheduler, metric):
        if self.current_epoch < 100:
            return
        else:
            super().lr_scheduler_step(scheduler, metric)
