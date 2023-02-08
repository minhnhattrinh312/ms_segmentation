import torch
import torch.nn as nn

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
        yTrueOnehot = torch.zeros(y_true.size(0), self.num_classes, y_true.size(2), y_true.size(3), y_true.size(4), device=self.device)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)
        y_pred = torch.clamp(y_pred,min=1e-5,max=1-1e-5)

        active_focal = - yTrueOnehot * (1-y_pred)**self.gamma * torch.log(y_pred) \
                        - (1 - yTrueOnehot) * y_pred**self.gamma * torch.log(1 - y_pred)
        loss = torch.sum(active_focal, dim=[2, 3, 4]) * self.class_weight
        return torch.sum(loss) / (torch.sum(self.class_weight) * y_true.size(0) * y_true.size(2) * y_true.size(3) * y_true.size(4))