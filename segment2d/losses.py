import torch
import torch.nn as nn


class ActiveFocalContourLoss(nn.Module):
    def __init__(self, device, class_weight, num_classes, gamma=2):
        """
        class weight should be a list.
        """
        super().__init__()
        self.device = device
        self.class_weight = torch.tensor(class_weight, device=device)
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = y_true.long()
        yTrueOnehot = torch.zeros(
            y_true.size(0),
            self.num_classes,
            y_true.size(2),
            y_true.size(3),
            device=self.device,
        )
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)
        y_pred = torch.clamp(y_pred, min=1e-5, max=1 - 1e-5)

        active_focal = -yTrueOnehot * (1 + (1 - y_pred) ** self.gamma) * torch.log(y_pred) - (1 - yTrueOnehot) * (1 + y_pred**self.gamma) * torch.log(1 - y_pred)
        active_focal = torch.sum(active_focal, dim=[2, 3]) * self.class_weight

        active_contour = yTrueOnehot * (1 - y_pred) + (1 - yTrueOnehot) * y_pred
        active_contour = torch.sum(active_contour, dim=[2, 3]) * self.class_weight

        loss = torch.sum(active_focal) + torch.sum(active_contour)
        return loss / (torch.sum(self.class_weight) * y_true.size(0) * y_true.size(2) * y_true.size(3))


class ActiveContourLoss(nn.Module):
    def __init__(self, device, class_weight, num_classes, gamma=2):
        """
        class weight should be a list.
        """
        super().__init__()
        self.device = device
        self.class_weight = torch.tensor(class_weight, device=device)
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = y_true.long()
        yTrueOnehot = torch.zeros(
            y_true.size(0),
            self.num_classes,
            y_true.size(2),
            y_true.size(3),
            device=self.device,
        )
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)
        y_pred = torch.clamp(y_pred, min=1e-5, max=1 - 1e-5)

        active_contour = yTrueOnehot * (1 - y_pred) + (1 - yTrueOnehot) * y_pred
        active_contour = torch.sum(active_contour, dim=[2, 3]) * self.class_weight

        loss = torch.sum(active_contour)
        return loss / (torch.sum(self.class_weight) * y_true.size(0) * y_true.size(2) * y_true.size(3))


class ActiveFocalLoss(nn.Module):
    def __init__(self, device, class_weight, num_classes, gamma=2):
        """
        class weight should be a list.
        """
        super().__init__()
        self.device = device
        self.class_weight = torch.tensor(class_weight, device=device)
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = y_true.long()
        yTrueOnehot = torch.zeros(
            y_true.size(0),
            self.num_classes,
            y_true.size(2),
            y_true.size(3),
            device=self.device,
        )
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)
        y_pred = torch.clamp(y_pred, min=1e-5, max=1 - 1e-5)

        active_focal = -yTrueOnehot * (1 + (1 - y_pred) ** self.gamma) * torch.log(y_pred) - (1 - yTrueOnehot) * (1 + y_pred**self.gamma) * torch.log(1 - y_pred)
        active_focal = torch.sum(active_focal, dim=[2, 3]) * self.class_weight

        loss = torch.sum(active_focal)
        return loss / (torch.sum(self.class_weight) * y_true.size(0) * y_true.size(2) * y_true.size(3))


class CrossEntropy(nn.Module):
    def __init__(self, device, num_classes):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = y_true.long()
        yTrueOnehot = torch.zeros(
            y_true.size(0),
            self.num_classes,
            y_true.size(2),
            y_true.size(3),
            device=self.device,
        )
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)

        loss = torch.sum(-yTrueOnehot * torch.log(y_pred + 1e-10))
        return loss / (y_true.size(0) * y_true.size(2) * y_true.size(3))


class DiceLoss(nn.Module):
    def __init__(self, device, num_classes):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = y_true.long()
        yTrueOnehot = torch.zeros(
            y_true.size(0),
            self.num_classes,
            y_true.size(2),
            y_true.size(3),
            device=self.device,
        )
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)[:, 1:]
        y_pred = y_pred[:, 1:]

        intersection = torch.sum(yTrueOnehot * y_pred, dim=[1, 2, 3])
        cardinality = torch.sum(yTrueOnehot + y_pred, dim=[1, 2, 3])
        loss = 1.0 - torch.mean((2.0 * intersection + 1e-5) / (cardinality + 1e-5))
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, device, num_classes, alpha=0.7):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = y_true.long()
        yTrueOnehot = torch.zeros(
            y_true.size(0),
            self.num_classes,
            y_true.size(2),
            y_true.size(3),
            device=self.device,
        )
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)[:, 1:]
        y_pred = y_pred[:, 1:]

        TP = torch.sum(yTrueOnehot * y_pred, dim=[1, 2, 3])
        FN = torch.sum(yTrueOnehot * (1 - y_pred), dim=[1, 2, 3])
        FP = torch.sum((1 - yTrueOnehot) * y_pred, dim=[1, 2, 3])
        loss = 1 - torch.mean((TP + 1e-5) / (TP + self.alpha * FN + (1 - self.alpha) * FP + 1e-5))
        return loss


class MSELoss(nn.Module):
    def __init__(self, device, num_classes, alpha=0.7):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = y_true.long()
        yTrueOnehot = torch.zeros(
            y_true.size(0),
            self.num_classes,
            y_true.size(2),
            y_true.size(3),
            device=self.device,
        )
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)[:, 1:]

        loss = torch.sum((yTrueOnehot - y_pred) ** 2)
        return loss / (y_true.size(0) * y_true.size(2) * y_true.size(3))
