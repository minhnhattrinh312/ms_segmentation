import torch
import numpy as np
from skimage import measure

def dice_MS(y_true, y_pred, smooth = 1e-5):
    output_standard = torch.argmax(y_pred, dim=1, keepdim=True)
    output = torch.where(output_standard == 1, 1, 0)
    label = torch.where(y_true == 1, 1, 0)

    intersection = torch.sum(label * output, dim=[1,2,3,4])
    union = torch.sum(label, dim=[1,2,3,4]) + torch.sum(output, dim=[1,2,3,4])
    return torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)

def dice_MS_volume(y_true, y_pred, smooth = 1e-5):
    y_pred = np.where(y_pred == 1, 1, 0)
    y_true = np.where(y_true == 1, 1, 0)
    intersection = np.sum(y_true * y_pred)
    cardinality  = np.sum(y_true + y_pred)
    return (2. * intersection + smooth) / (cardinality + smooth)

def seg_metrics(y_true,  y_pred):
    seg_total = np.sum(y_pred)
    truth_total = np.sum(y_true)
    tp = np.sum(y_pred[y_true == 1])
    dice = 2 * tp / (seg_total + truth_total)
    ppv = tp / (seg_total + 0.001)
    tpr = tp / (truth_total + 0.001)
    vd = abs(seg_total - truth_total) / truth_total

    # calculate LFPR
    seg_labels, seg_num = measure.label(y_pred, return_num=True, connectivity=2)
    lfp_cnt = 0
    tmp_cnt = 0
    for label in range(1, seg_num + 1):
        tmp_cnt += np.sum(y_pred[seg_labels == label])
        if np.sum(y_true[seg_labels == label]) == 0:
            lfp_cnt += 1
    lfpr = lfp_cnt / (seg_num + 0.001)

    # calculate LTPR
    truth_labels, truth_num = measure.label(y_true, return_num=True, connectivity=2)
    ltp_cnt = 0
    for label in range(1, truth_num + 1):
        if np.sum(y_pred[truth_labels == label]) > 0:
            ltp_cnt += 1
    ltpr = ltp_cnt / truth_num

    return {"dice": dice, "ppv": ppv, "tpr": tpr, "vd": vd, "lfpr": lfpr, "ltpr": ltpr}