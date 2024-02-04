
# Path: segment2d/config.py
from yacs.config import CfgNode as CN

cfg = CN()
cfg.DATA = CN()
cfg.TRAIN = CN()
cfg.SYS = CN()
cfg.OPT = CN()
cfg.DIRS = CN()
cfg.PREDICT = CN()
cfg.DIRS.PCL = CN()


cfg.DATA.NUM_CLASS = 2
cfg.DATA.CLASS_WEIGHT = [0.05, 0.95] #default [0.1, 0.9]
cfg.DATA.DIM2PAD_ISBI = [224, 224, 224]
cfg.DATA.INDIM_MODEL = 4
cfg.DATA.INDIM_MODEL_MICCAI = 5
cfg.DATA.INDIM_MODEL_MICCAI2008 = 3
cfg.DATA.DIM2PAD_MICCAI = [256, 256, 256]
cfg.DATA.DIM2PAD_MICCAI2008 = [320, 320, 320]
cfg.DATA.ORIGIN2CUT = [(1, 2, 0), (0, 2, 1), (0, 1, 2)]
cfg.DATA.CUT2ORIGIN = [(2, 0, 1), (0, 2, 1), (0, 1, 2)]
# cfg.DATA.ORIGIN2CUT = [(0, 1, 2)]
# cfg.DATA.CUT2ORIGIN = [(0, 1, 2)]

cfg.TRAIN.TASK = "msseg2008" # "isbi" or msseg or msseg2008
# "active_focal" or "focal_contour" or "active_contour"
#TverskyLoss,  CrossEntropy, DiceLoss, MSELoss
cfg.TRAIN.LOSS = "focal_contour" 

cfg.TRAIN.DISTINCT_SUBJECT = True
cfg.TRAIN.FREEZE = True
cfg.TRAIN.EVA_N_EPOCHS = 2
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.EPOCHS = 1000
cfg.TRAIN.NUM_WORKERS = 8
cfg.TRAIN.PREFETCH_FACTOR = 4
cfg.TRAIN.FOLD = 1
cfg.TRAIN.LOAD_CHECKPOINT = True
cfg.TRAIN.SAVE_TOP_K = 5
cfg.TRAIN.IDX_CHECKPOINT = -1
cfg.TRAIN.WANDB = True
cfg.TRAIN.AUGMENTATION = True
cfg.TRAIN.NORMALIZE = "min_max" # "min_max"
cfg.TRAIN.MODALITIES = False
cfg.TRAIN.PRETRAIN = "imagenet" #  "pcl" or imagenet

# adjust input dim of the model
cfg.DATA.DIM_SIZE = 12 if cfg.TRAIN.MODALITIES else 3

cfg.SYS.ACCELERATOR = "gpu"
cfg.SYS.DEVICES = [0]
cfg.SYS.MIX_PRECISION = 16 #32 or 16

cfg.OPT.LEARNING_RATE = 0.0002
cfg.OPT.FACTOR_LR = 0.5
cfg.OPT.PATIENCE_LR = 50
cfg.OPT.PATIENCE_ES = 270

cfg.DIRS.PCL.SAVE_CONVNEXT = "./weights_pcl_convnext/"
# cfg.DIRS.PCL.SAVE_RESNET50 = "./weights_pcl_resnet50_focal/"
cfg.DIRS.PCL.SAVE_RESNET50 = f"./weights_resnet50_{cfg.TRAIN.LOSS}/"
# cfg.DIRS.PREDICT_DIR = "./predict_testset/"
cfg.DIRS.PREDICT_DIR = "./nhatvinbig/"

# cfg.PREDICT.IDX_CHECKPOINT = -1
cfg.PREDICT.BATCH_SIZE = 8
cfg.PREDICT.MIN_SIZE_REMOVE = 3
cfg.PREDICT.MODE = "2D" # "3D"
cfg.PREDICT.ENSEMBLE = False
cfg.PREDICT.MASK_EXIST = False
cfg.PREDICT.MODEL = "tiramisu" # "tiramisu"or resnet50
# cfg.PREDICT.NAME_ZIP = f"{cfg.PREDICT.MODEL}_95900195_{cfg.PREDICT.MIN_SIZE_REMOVE}" # or resnet50
cfg.PREDICT.NAME_ZIP = "vinbigdata_nhat"
cfg.PREDICT.WEIGHTS = [0.9,0.84,0.01,1,0.95]
# cfg.PREDICT.WEIGHTS = [0.9,0.84,0.01,1,0.95]
# 93.255
# 0.999452178396494	0.998410243189825	0.01 1	0.999097705594225

if cfg.TRAIN.TASK == "isbi":
    cfg.TRAIN.EPOCH_WARMUP = 50
elif cfg.TRAIN.TASK == "msseg":
    cfg.TRAIN.EPOCH_WARMUP = 30

cfg.DIRS.SAVE_DIR = f"./weights_{cfg.TRAIN.TASK}_{cfg.PREDICT.MODEL}/"
    