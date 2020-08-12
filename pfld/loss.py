import torch
from torch import nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize):
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        l2_distant = torch.sum((landmark_gt - landmarks) ** 2, axis=1)
        return torch.mean(weight_angle * l2_distant), torch.mean(l2_distant)


def SmoothL1(y_true, y_pred, beta=1):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum( torch.where(mae > beta, mae - 0.5 * beta, 0.5 * mae ** 2 / beta), axis=-1)
    return torch.mean(loss)


# class WingLoss(nn.Module):
#     def __init__(self, omega=10, epsilon=2):
#         super(WingLoss, self).__init__()
#         self.omega = omega
#         self.epsilon = epsilon
#
#     def forward(self, pred, target):
#         y = target
#         y_hat = pred
#         delta_y = (y - y_hat).abs()
#         delta_y1 = delta_y[delta_y < self.omega]
#         delta_y2 = delta_y[delta_y >= self.omega]
#         loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
#         C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
#         loss2 = delta_y2 - C
#         return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0, N_LANDMARK=106):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.N_LANDMARK = N_LANDMARK

    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1, self.N_LANDMARK, 2)
        y_true = y_true.reshape(-1, self.N_LANDMARK, 2)

        x = y_true - y_pred
        c = self.w * (1.0 - math.log(1.0 + self.w / self.epsilon))
        absolute_x = torch.abs(x)
        losses = torch.where(
            self.w > absolute_x,
            self.w * torch.log(1.0 + absolute_x / self.epsilon),
            absolute_x - c)
        loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
        return loss, loss


# def WingLoss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK=106):
#     y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
#     y_true = y_true.reshape(-1, N_LANDMARK, 2)
#
#     x = y_true - y_pred
#     c = w * (1.0 - math.log(1.0 + w / epsilon))
#     absolute_x = torch.abs(x)
#     losses = torch.where(
#         w > absolute_x,
#         w * torch.log(1.0 + absolute_x / epsilon),
#         absolute_x - c)
#     loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
#     return loss
