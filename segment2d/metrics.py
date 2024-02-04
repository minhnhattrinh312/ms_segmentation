import torch
import numpy as np
from skimage import measure
#import OrderedDict
# import Pearson's correlation coefficient
from scipy.stats import pearsonr
from skimage import measure

from collections import OrderedDict
def dice_MS(y_true, y_pred, smooth = 1e-5):
    output_standard = torch.argmax(y_pred, dim=1, keepdim=True)
    output = torch.where(output_standard == 1, 1, 0)
    label = torch.where(y_true == 1, 1, 0)

    intersection = torch.sum(label * output, dim=[1,2,3])
    union = torch.sum(label, dim=[1,2,3]) + torch.sum(output, dim=[1,2,3])
    return torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)

def dice_MS_volume(y_true, y_pred, smooth = 1e-5):
    y_pred = np.where(y_pred == 1, 1, 0)
    y_true = np.where(y_true == 1, 1, 0)
    intersection = np.sum(y_true * y_pred)
    cardinality  = np.sum(y_true + y_pred)
    return (2. * intersection + smooth) / (cardinality + smooth)

def seg_metrics(truth_vol, seg_vol, output_errors=False):
    seg_total = np.sum(seg_vol)
    truth_total = np.sum(truth_vol)
    tp = np.sum(seg_vol[truth_vol == 1])
    dice = (2 * tp + 0.0001) / (seg_total + truth_total + 0.0001)
    ppv = (tp + 0.0001) / (seg_total + 0.0001)
    tpr = (tp + 0.0001) / (truth_total + 0.0001)
    vd = abs(seg_total - truth_total) / (truth_total + 0.0001)

    # calculate LFPR
    seg_labels, seg_num = measure.label(seg_vol, return_num=True, connectivity=2)
    lfp_cnt = 0
    tmp_cnt = 0
    for label in range(1, seg_num + 1):
        tmp_cnt += np.sum(seg_vol[seg_labels == label])
        if np.sum(truth_vol[seg_labels == label]) == 0:
            lfp_cnt += 1
    lfpr = (lfp_cnt + 0.0001)/ (seg_num + 0.0001)

    # calculate LTPR
    truth_labels, truth_num = measure.label(truth_vol, return_num=True, connectivity=2)
    ltp_cnt = 0
    for label in range(1, truth_num + 1):
        if np.sum(seg_vol[truth_labels == label]) > 0:
            ltp_cnt += 1
    ltpr = (ltp_cnt + 0.0001) / (truth_num+0.0001)

    # calculate Pearson's correlation coefficient
    corr = pearsonr(seg_vol.flatten(), truth_vol.flatten())[0]
    # print("Timed used calculating metrics: ", time.time() - time_start)

    # calculate score following isbi 2015 challenge
    isbi_score = dice/8 + ppv/8 + (1-lfpr)/4 + ltpr/4 + corr/4
    # isbi_score = dice + ppv + (1 - lfpr) * 2 + ltpr * 2
    metrics = dict()
    metrics["dice"] = dice
    metrics["ppv"] = ppv
    metrics["tpr"] = tpr
    metrics["lfpr"] = lfpr
    metrics["ltpr"] = ltpr
    metrics["vd"] = vd
    metrics["isbi_score"] = isbi_score
    return metrics

def seg_metrics_miccai2008(truth_vol, seg_vol, output_errors=False):
    seg_total = np.sum(seg_vol)
    truth_total = np.sum(truth_vol)
    tp = np.sum(seg_vol[truth_vol == 1])
    fp = np.sum(seg_vol[truth_vol == 0])
    # dice = (2 * tp + 0.0001) / (seg_total + truth_total + 0.0001)
    tpr = (tp + 0.0001) / (truth_total + 0.0001)
    fpr = (fp + 0.0001) / (fp + tp + 0.0001)
    vd = abs(seg_total - truth_total) / (truth_total + 0.0001)

    # calculate score following isbi 2015 challenge
    # score = dice/8 + ppv/8 + (1-lfpr)/4 + ltpr/4 + corr/4
    score = 100*(tpr - fpr - vd / 8)
    metrics = dict()
    # metrics["dice"] = dice
    metrics["tpr"] = tpr
    metrics["fpr"] = fpr
    metrics["vd"] = vd
    metrics["score"] = score
    return metrics