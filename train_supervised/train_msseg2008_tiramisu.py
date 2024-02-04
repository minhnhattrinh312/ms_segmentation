import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "/home/nhattm1/test_newcode")
# from segment3d import ISBILoader, RandomFlip, cfg, SkipNet, Segmenter
from segment2d import *
from pytorch_lightning.loggers import WandbLogger
import os
import csv
import wandb

subjects_dir = "data/msseg2008npz/"
subjects_nii_dir = "data/msseg-2008-training-nii/"
name_flair = "FLAIR_brain_bias_correction.nii.gz"
torch.set_float32_matmul_precision('high')
with open('subject_msseg2008.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    # Read the rows of the file
    rows = list(reader)

# Main function
if __name__ == "__main__":
    # Loop over the folds
    for i in range(4, 6):
        cfg.TRAIN.FOLD = i
        print("train on fold", cfg.TRAIN.FOLD)
        cfg.DIRS.SAVE_DIR = f"./weights_msseg2008_{cfg.PREDICT.MODEL}/fold{cfg.TRAIN.FOLD}/"
        model = FCDenseNet(in_channels=cfg.DATA.INDIM_MODEL_MICCAI2008, n_classes=cfg.DATA.NUM_CLASS)

        # create folder to save checkpoints
        os.makedirs(cfg.DIRS.SAVE_DIR, exist_ok=True)
        # List of subjects in the training set
        list_train_subject = []
        segmenter = Segmenter(model, cfg.DATA.CLASS_WEIGHT, cfg.DATA.NUM_CLASS, 
                                                cfg.OPT.LEARNING_RATE, cfg.OPT.FACTOR_LR, cfg.OPT.PATIENCE_LR)
        ids_for_test = []
        for row in rows:
            name_subject_npz = row['name']
            name_subject = row['subject']
            fold = int(row['fold'])
            if fold != cfg.TRAIN.FOLD:
                list_train_subject.append(f"{subjects_dir}{name_subject_npz}.npz")
            elif name_subject not in ids_for_test:
                ids_for_test.append(name_subject)
        print("ids_for_test:", ids_for_test)
        list_test_subject = [f"{subjects_nii_dir}{name_subject}/{name_subject}_{name_flair}" \
                             for name_subject in ids_for_test]
        # Create a ISBILoader object for the training data using the list_train_subject variable and the defined augmentations
        train_data = MSSEGLoader(list_subject=list_train_subject)

        # Create a ISBILoader object for the test data using the list_test_subject variable

        test_dataset = MSSEG2008_Test_Loader(list_test_subject)

        # If wandb_logger is True, create a WandbLogger object
        if cfg.TRAIN.WANDB:
            wandb_logger = WandbLogger(project="msseg2008", group=f"tiramisu_{cfg.TRAIN.LOSS}", name=f"fold{cfg.TRAIN.FOLD}_{cfg.TRAIN.LOSS}", resume="allow")
        else:
            wandb_logger = False
        # Define data loaders for the training and test data
        train_dataset = DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, pin_memory=True,
                            shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS,
                            drop_last=True, prefetch_factor = cfg.TRAIN.PREFETCH_FACTOR)

        # Initialize a ModelCheckpoint callback to save the model weights after each epoch
        check_point_score = pl.callbacks.model_checkpoint.ModelCheckpoint(cfg.DIRS.SAVE_DIR, 
            filename="ckpt_score_{val_score:0.4f}", monitor="val_score", mode="max", save_top_k=cfg.TRAIN.SAVE_TOP_K,
            verbose=True, save_weights_only=True, auto_insert_metric_name=False, save_last=True)
        
        check_point_tpr = pl.callbacks.model_checkpoint.ModelCheckpoint(cfg.DIRS.SAVE_DIR, 
            filename="ckpt_tpr_{val_tpr:0.4f}", monitor="val_tpr", mode="max", save_top_k=cfg.TRAIN.SAVE_TOP_K,
            verbose=True, save_weights_only=True, auto_insert_metric_name=False, save_last=True)
        # Initialize a LearningRateMonitor callback to log the learning rate during training
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # Initialize a EarlyStopping callback to stop training if the validation loss does not improve for a certain number of epochs
        early_stopping = EarlyStopping(monitor="val_score", mode="max", patience=cfg.OPT.PATIENCE_ES, verbose=True, strict=False)
            
        print("class_weight:", cfg.DATA.CLASS_WEIGHT)
        print("Train on fold:", cfg.TRAIN.FOLD)
        print("Use loss:", cfg.TRAIN.LOSS)

        # Define a dictionary with the parameters for the Trainer object
        PARAMS_TRAINER = {"accelerator":cfg.SYS.ACCELERATOR, "devices":cfg.SYS.DEVICES, 
                        "benchmark":True, "enable_progress_bar":True, 
                            # "overfit_batches" :5,
                            "logger" : wandb_logger,
                            "callbacks" : [check_point_score, check_point_tpr, lr_monitor, early_stopping],
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
