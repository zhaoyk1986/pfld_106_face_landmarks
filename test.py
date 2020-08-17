# ------------------------------------------------------------------------------
# Copyright (c) Zhichao Zhao
# Licensed under the MIT License.
# Created by Zhichao zhao(zhaozhichao4515@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import time
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import PFLDDatasets

# from models.mobilev3_pfld import PFLDInference

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 106:
            interocular = np.linalg.norm(pts_gt[35, ] - pts_gt[93, ])  # 左眼角和右眼角的欧式距离
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / interocular

    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate 


def validate(wlfw_val_dataloader, plfd_backbone):
    plfd_backbone.eval()

    nme_list = []
    cost_time = []
    with torch.no_grad():
        for img, landmark_gt, _ in wlfw_val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)

            start_time = time.time()
            _, landmarks = plfd_backbone(img)
            cost_time.append(time.time() - start_time)

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)  # landmark
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()  # landmark_gt

            if args.show_image:
                show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                pre_landmark = landmarks[0] * [112, 112]

                cv2.imwrite("xxx.jpg", show_img)
                img_clone = cv2.imread("xxx.jpg")

                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(img_clone, (x, y), 1, (255, 0, 0),-1)
                cv2.imshow("xx.jpg", img_clone)
                cv2.waitKey(0)

            nme_temp = compute_nme(landmarks, landmark_gt)
            for item in nme_temp:
                nme_list.append(item)

        # nme
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        # auc and failure rate
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold, False)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))
        # inference time
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))


def main(args):
    if args.backbone == "v2":
        from models.pfld import PFLDInference, AuxiliaryNet
    elif args.backbone == "v3":
        from models.mobilev3_pfld import PFLDInference, AuxiliaryNet
    elif args.backbone == "ghost":
        from models.ghost_pfld import PFLDInference, AuxiliaryNet
    elif args.backbone == "lite":
        from models.lite import PFLDInference, AuxiliaryNet
    else:
        raise ValueError("backbone is not implemented")

    checkpoint = torch.load(args.model_path, map_location=device)
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'], strict=False)

    transform = transforms.Compose([transforms.ToTensor()])
    wlfw_val_dataset = PFLDDatasets(args.test_dataset, transform, img_root=os.path.realpath('./data'))
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset, batch_size=1, shuffle=False, num_workers=0)

    validate(wlfw_val_dataloader, plfd_backbone)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--backbone', default='v2', type=str, choices=["v2", "v3", "lite", "ghost"])
    parser.add_argument('--model_path', default="", type=str)
    parser.add_argument('--test_dataset', default='./data/test_data/list.txt', type=str)
    parser.add_argument('--show_image', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
