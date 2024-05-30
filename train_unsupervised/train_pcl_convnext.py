import sys

sys.path.insert(0, "/home/nhattm1/test_newcode")
from PCL import *
from tqdm import tqdm
from timm.optim import Nadam
import torch
from tqdm import tqdm
import wandb
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# resnet = models.resnet50(pretrained=True)
os.makedirs(cfg.DIRS.SAVE_CONVNEXT, exist_ok=True)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ConvNextEncoder(in_dim=cfg.DATA.DIM_SIZE, num_class=cfg.DATA.NUM_CLASS)
checkpoint = torch.hub.load_state_dict_from_url(url=convnext_urls["convnext_tiny_1k"], map_location="cpu", check_hash=True)
model.load_state_dict(checkpoint["model"], strict=False)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! load pretrained model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

if cfg.TRAIN.LOAD_CKPT:
    # load checkpoint
    try:
        checkpoint = torch.load(cfg.DIRS.SAVE_CONVNEXT + "best_model-0.7038.pt")
        model.load_state_dict(checkpoint)
        print("load last model")
    except:
        print("no checkpoint found")

learner = PCL_ConvNext(
    model,
    image_size=224,
    hidden_layer_pixel="stages",  # leads to output of 8x8 feature map for pixel-level learning
    hidden_layer_instance="final_layer",  # leads to output for instance-level learning
    projection_size=256,  # size of projection output, 256 was used in the paper
    projection_hidden_size=2048,  # size of projection hidden dimension, paper used 2048
    moving_average_decay=0.99,  # exponential moving average decay of target encoder
    ppm_num_layers=1,  # number of layers for transform function in the pixel propagation module, 1 was optimal
    ppm_gamma=2,  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
    distance_thres=0.7,  # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
    similarity_temperature=0.3,  # temperature for the cosine similarity for the pixel contrastive loss
    alpha=1.0,  # weight of the pixel propagation loss (pixpro) vs pixel CL loss
    use_pixpro=True,  # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
    cutout_ratio_range=(
        0.6,
        0.8,
    ),  # a random ratio is selected from this range for the random cutout
)
learner = learner.to(device)

optimizer = Nadam(learner.parameters(), lr=1e-4)


# Initialize wandb
wandb.init(project="pcl", name="convnext_pcl_2", resume="allow")

dataset = BratsLoader(train_path="./data/brats/*npz/*")
# Define your learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=True)

# Set up some variables to track the training progress

steps_per_epoch = len(dataset) // cfg.TRAIN.BATCH_SIZE
global_step = 0
running_loss_1000 = 0
best_mean_loss = float("inf")
num_epochs = cfg.TRAIN.EPOCHS
early_stopping_counter = 0
patience_epoch = 35
save_dir = cfg.DIRS.SAVE_CONVNEXT
data_loader = DataLoader(
    dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    pin_memory=True,
    shuffle=True,
    num_workers=cfg.TRAIN.NUM_WORKERS,
    drop_last=True,
    prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
)
if __name__ == "__main__":
    # Start the training loop
    for epoch in range(num_epochs):
        running_loss = 0
        for images in tqdm(data_loader):
            images = images.to(device).float()
            loss = learner(images)  # if positive pixel pairs is equal to zero, the loss is equal to the instance level loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
            # Update the running loss and global step
            running_loss += loss.item()
            running_loss_1000 += loss.item()
            global_step += 1

            # Print the loss every 100 steps
            if global_step % 1000 == 0:
                print(f"Step [{global_step}/{num_epochs*steps_per_epoch}] Loss: {running_loss_1000/1000}")
                wandb.log(
                    {
                        "Loss": running_loss_1000 / 1000,
                        "Learning Rate": optimizer.param_groups[0]["lr"],
                    }
                )
                running_loss_1000 = 0

        if (epoch + 1) % 1 == 0:
            # remove the previous epoch model
            path_previous_epoch = f"{save_dir}weight_{epoch-1}.pt"
            if os.path.exists(path_previous_epoch):
                os.remove(path_previous_epoch)
            torch.save(model.state_dict(), f"{save_dir}weight_{epoch}.pt")
        # Calculate the mean loss for the epoch
        mean_loss = running_loss / steps_per_epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] Mean Loss: {mean_loss}")
        wandb.log(
            {
                "Mean Loss": mean_loss,
                "Learning Rate": optimizer.param_groups[0]["lr"],
                "Epoch": epoch,
            }
        )

        # Check if the mean loss is the best seen so far
        if mean_loss < best_mean_loss:
            path_best_mean_loss = f"{save_dir}best_model{best_mean_loss:0.4f}.pt"
            print("mean loss improved, from {} to {}, saving best model...".format(best_mean_loss, mean_loss))
            # remove the previous best model
            if os.path.isfile(path_best_mean_loss):
                os.remove(path_best_mean_loss)

            best_mean_loss = mean_loss
            torch.save(model.state_dict(), f"{save_dir}best_model{mean_loss:0.4f}.pt")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        # Check if the mean loss hasn't improved for 3 epochs, and reduce the learning rate if so
        scheduler.step(mean_loss)

        if early_stopping_counter >= patience_epoch:
            print(f"Validation Loss has not improved for {patience_epoch} epochs, stopping early...")
            break

    wandb.finish()
