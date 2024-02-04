import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "/home/nhattm1/test_newcode")
# from segment3d import EGLoader, RandomFlip, cfg, SkipNet, Segmenter
from segment2d import *
from pytorch_lightning.loggers import WandbLogger
import os
import csv
import wandb
import segmentation_models_pytorch as smp
subjects_dir = "./data/data_isbi_2015/isbi2npz2D/"
torch.set_float32_matmul_precision('high')


# Main function
if __name__ == "__main__":
    model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=cfg.DATA.INDIM_MODEL_MICCAI,
            classes=2,
            activation = "softmax2d"
        )
    if cfg.TRAIN.PRETRAIN == "pcl":
        print("use self supervised model")
        model.encoder.load_state_dict(torch.load(cfg.DIRS.PCL.SAVE_RESNET50 + "best_model.pt"))
    else:
        print("use imagenet pretrained model")

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! load pretrained model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    # create folder to save checkpoints
    os.makedirs(cfg.DIRS.SAVE_DIR, exist_ok=True)
    # List of subjects in test set
    list_test_subject = sorted(glob.glob(f"data/MSSEG-Testing/Center*/*/Preprocessed_Data/*FLAIR*"))

    # List of subjects in the training set
    list_train_subject = sorted(glob.glob("data/msseg2016npz/*"))
    
    # Create a MEGLoader object for the training data using the list_train_subject variable and the defined augmentations
    train_data = MSSEGLoader(list_subject=list_train_subject)

    # Create a MEGLoader object for the test data using the list_test_subject variable
    # test_data = MSEGLoader(list_subject=list_test_subject)
    test_dataset = MSSEG_Test_Loader(list_test_subject)

    # If wandb_logger is True, create a WandbLogger object
    if cfg.TRAIN.WANDB:
        wandb_logger = WandbLogger(project="msseg", group="resnet50", name=f"resnet50_supervised", resume="allow")
    else:
        wandb_logger = False
    # Define data loaders for the training and test data
    train_dataset = DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True,
                        shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS,
                        drop_last=True, prefetch_factor = cfg.TRAIN.PREFETCH_FACTOR)
    # test_dataset = DataLoader(test_data, batch_size=cfg.TRAIN.BATCH_SIZE,
    #                         num_workers=cfg.TRAIN.NUM_WORKERS, prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR)

    # Initialize the segmentation model with the specified parameters
    segmenter = Segmenter(model, cfg.DATA.CLASS_WEIGHT, cfg.DATA.NUM_CLASS, 
                                            cfg.OPT.LEARNING_RATE, cfg.OPT.FACTOR_LR, cfg.OPT.PATIENCE_LR)


    check_point_dice = pl.callbacks.model_checkpoint.ModelCheckpoint(cfg.DIRS.SAVE_DIR, 
        filename="ckpt_dice_{test_dice:0.4f}", monitor="test_dice", mode="max", save_top_k=cfg.TRAIN.SAVE_TOP_K,
        verbose=True, save_weights_only=True, auto_insert_metric_name=False, save_last=True)
    # Initialize a LearningRateMonitor callback to log the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping(monitor="test_dice", mode="max", patience=cfg.OPT.PATIENCE_ES, verbose=True, strict=False)
    print("class_weight:", cfg.DATA.CLASS_WEIGHT)

        
    segmenter.model.encoder.requires_grad_(True)
    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {"accelerator":cfg.SYS.ACCELERATOR, "devices":cfg.SYS.DEVICES, 
                    "benchmark":True, "enable_progress_bar":True, 
                        # "overfit_batches" :5,
                        "logger" : wandb_logger,
                        "callbacks" : [check_point_dice, lr_monitor, early_stopping],
                        "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":cfg.TRAIN.EPOCHS,
                        "precision":cfg.SYS.MIX_PRECISION,
                        }

    # Initialize a Trainer object with the specified parameters
    trainer = pl.Trainer(**PARAMS_TRAINER)
    # Get a list of file paths for all non-hidden files in the SAVE_DIR directory
    checkpoint_paths = [cfg.DIRS.SAVE_DIR+f for f in os.listdir(cfg.DIRS.SAVE_DIR) if not f.startswith('.')]
    checkpoint_paths.sort()
    # If there are checkpoint paths and the load_checkpoint flag is set to True
    if checkpoint_paths and cfg.TRAIN.LOAD_CHECKPOINT:
        # Select the second checkpoint in the list (index 0)
        checkpoint = checkpoint_paths[cfg.TRAIN.IDX_CHECKPOINT]
        print(f"load checkpoint: {checkpoint}")
        # Load the model weights from the selected checkpoint
        segmenter = Segmenter.load_from_checkpoint(checkpoint_path=checkpoint, model=model,
                                                class_weight=cfg.DATA.CLASS_WEIGHT,
                                                    num_classes=cfg.DATA.NUM_CLASS, 
                                                learning_rate=cfg.OPT.LEARNING_RATE,
                                                factor_lr=cfg.OPT.FACTOR_LR, patience_lr=cfg.OPT.PATIENCE_LR)
    

    # Train the model using the train_dataset and test_dataset data loaders
    trainer.fit(segmenter, train_dataset, test_dataset)
    
    wandb.finish()
