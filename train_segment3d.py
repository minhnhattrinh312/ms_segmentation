import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch.utils.data import DataLoader
from segment3d import ISBILoader, RandomFlip, config, SkipNet, Segmentor
from pytorch_lightning.loggers import WandbLogger
import random
import glob
import os
import numpy as np
import csv
# Main function
if __name__ == "__main__":
    # Load a list of filenames using glob.glob and sort it
    subjects_dir = "./data/data_isbi_2015/isbi2npz3D/"
    with open('subject2name.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # Read the rows of the file
        rows = list(reader)

    # List of subjects in test set
    list_test_subject = []
    # List of subjects in the traing set
    list_train_subject = []

    for row in rows:
        name_subject = row['name']
        fold = int(row['fold'])
        if fold == config["TRAIN"]["FOLD"]:
            list_test_subject.append(f"{subjects_dir}{name_subject}.npz")
        else:
            list_train_subject.append(f"{subjects_dir}{name_subject}.npz")


    # Define a list of augmentations to be applied to the training data
    if config["TRAIN"]["AUGMENTATION"]:
        augmentation = [
            RandomFlip(0.5)
            ]
    else:
        augmentation = None
    
    # Create a ISBILoader object for the training data using the list_train_subject variable and the defined augmentations
    train_data = ISBILoader(list_subject=list_train_subject, transform=augmentation)
    
    # Create a ISBILoader object for the test data using the list_test_subject variable
    test_data = ISBILoader(list_subject=list_test_subject)
    
    # If wandb_logger is True, create a WandbLogger object
    if config["TRAIN"]["WANDB"]:
        wandb_logger = WandbLogger(project="segment3d")
    else:
        wandb_logger = False
    # Define data loaders for the training and test data
    train_dataset = DataLoader(train_data, batch_size=config["TRAIN"]["BATCH_SIZE"], pin_memory=True,
                        shuffle=True, num_workers=config["TRAIN"]["NUM_WORKERS"],
                        drop_last=True, prefetch_factor = config["TRAIN"]["PREFETCH_FACTOR"])
    test_dataset = DataLoader(test_data, batch_size=config["TRAIN"]["BATCH_SIZE"],
                          num_workers=config["TRAIN"]["NUM_WORKERS"], prefetch_factor=config["TRAIN"]["PREFETCH_FACTOR"])


    # define model
    model = SkipNet(in_dim = config["DATA"]["DIM_SIZE"], num_class=config["DATA"]["NUM_CLASS"])
    
    # create folder to save checkpoints
    os.makedirs(config["DIRS"]["SAVE_DIR"], exist_ok=True)
    
    # Initialize the segmentation model with the specified parameters
    segmentor = Segmentor(model, config["DATA"]["CLASS_WEIGHT"], config["DATA"]["NUM_CLASS"], 
                                            config["OPT"]["LEARNING_RATE"], config["OPT"]["FACTOR_LR"], config["OPT"]["PATIENCE_LR"])

    # Initialize a ModelCheckpoint callback to save the model weights after each epoch
    check_point = pl.callbacks.model_checkpoint.ModelCheckpoint(config["DIRS"]["SAVE_DIR"], 
        filename="ckpt{diceVal:0.4f}", monitor="diceVal", mode="max", save_top_k=config["TRAIN"]["SAVE_TOP_K"],
        verbose=True, save_weights_only=True, auto_insert_metric_name=False,)

    # Initialize a LearningRateMonitor callback to log the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {"accelerator":config["SYS"]["ACCELERATOR"], "devices":config["SYS"]["DEVICES"], 
                    "benchmark":True, "enable_progress_bar":True, 
                    #    "overfit_batches" :1,
                        "logger" : wandb_logger,
                        "callbacks" : [check_point, lr_monitor],
                        "log_every_n_steps" :1, "num_sanity_val_steps":1, "max_epochs":config["TRAIN"]["EPOCHS"],
                        "precision":config["SYS"]["MIX_PRECISION"],
                        }

    # Initialize a Trainer object with the specified parameters
    trainer = pl.Trainer(**PARAMS_TRAINER)
    print("class_weight:", config["DATA"]["CLASS_WEIGHT"])
    print("Train on fold:", config["TRAIN"]["FOLD"])
    # Get a list of file paths for all non-hidden files in the SAVE_DIR directory
    checkpoint_paths = [config["DIRS"]["SAVE_DIR"]+f for f in os.listdir(config["DIRS"]["SAVE_DIR"]) if not f.startswith('.')]

    # Set a flag to determine whether to load a checkpoint
    load_checkpoint = config["TRAIN"]["LOAD_CHECKPOINT"]

    # If there are checkpoint paths and the load_checkpoint flag is set to True
    if checkpoint_paths and load_checkpoint:
        # Select the second checkpoint in the list (index 0)
        checkpoint = checkpoint_paths[config["TRAIN"]["IDX_CHECKPOINT"]]
        print(f"load checkpoint: {checkpoint}")
        # Load the model weights from the selected checkpoint
        segmentor = Segmentor.load_from_checkpoint(checkpoint_path=checkpoint, model=model,
                                                class_weight=config["DATA"]['CLASS_WEIGHT'],
                                                    num_classes=config["DATA"]["NUM_CLASS"], 
                                                learning_rate=config["OPT"]["LEARNING_RATE"],
                                                factor_lr=config["OPT"]["FACTOR_LR"], patience_lr=config["OPT"]["PATIENCE_LR"])

    
    # Train the model using the train_dataset and test_dataset data loaders
    trainer.fit(segmentor, train_dataset, test_dataset)