import argparse
import logging
from pathlib import Path
import os

import numpy as np
import torch

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from dataset.datasets import PFLDDatasets
from pfld.loss import PFLDLoss as LandMarkLoss
from pfld.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, plfd_backbone, auxiliarynet, criterion, optimizer,
          epoch, log_interval=10):
    losses = AverageMeter()
    plfd_backbone.train()
    auxiliarynet.train()

    # print('is_training:', plfd_backbone.training)
    logging.info("total iteration is {}".format(len(train_loader)))
    weighted_loss, loss = 0, 0
    for iteration, (img, landmark_gt, euler_angle_gt) in enumerate(train_loader):
        img = img.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        plfd_backbone = plfd_backbone.to(device)
        auxiliarynet = auxiliarynet.to(device)

        features, landmarks = plfd_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(landmark_gt, euler_angle_gt,
                                        angle, landmarks, args.train_batchsize)
        # weighted_loss, loss = criterion(landmarks, landmark_gt)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        losses.update(loss.item())
        if iteration % log_interval == 0:
            logging.info("epoch: {}, iteration: {}, train loss: {:.4f}".format(epoch, iteration, weighted_loss.item()))
    return weighted_loss, loss


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
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion):
    plfd_backbone.eval()
    auxiliarynet.eval() 
    losses = []
    nme = []
    with torch.no_grad():
        for img, landmark_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = plfd_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))

            landmark = landmark.cpu().numpy().reshape(landmark.shape[0], -1, 2)
            landmark_gt = landmark_gt.cpu().numpy().reshape(landmark_gt.shape[0], -1, 2)
            nme_i = compute_nme(landmark, landmark_gt)
            losses.append(loss.cpu().numpy())
            for item in nme_i:
                nme.append(item)
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} nme: {:.4f}'.format(np.mean(losses), np.mean(nme)))
    return np.mean(losses), np.mean(nme)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(args.log_file, mode='w'),
                                  logging.StreamHandler()
                                  ]
                        )
    print_args(args)
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
    plfd_backbone = PFLDInference()
    auxiliarynet = AuxiliaryNet()
    if os.path.exists(args.resume) and args.resume.endswith('.pth'):
        logging.info("loading the checkpoint from {}".format(args.resume))
        check = torch.load(args.resume, map_location=torch.device('cpu'))
        plfd_backbone.load_state_dict(check["plfd_backbone"])
        auxiliarynet.load_state_dict(check["auxiliarynet"])
        args.start_epoch = check["epoch"]

    # Step 2: model, criterion, optimizer, scheduler
    plfd_backbone = plfd_backbone.to(device)
    auxiliarynet = auxiliarynet.to(device)
    criterion = LandMarkLoss()
    optimizer = torch.optim.Adam(
        [{
            'params': plfd_backbone.parameters()
        }, {
            'params': auxiliarynet.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=args.lr_patience, verbose=True)

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    wlfwdataset = PFLDDatasets(args.dataroot, transform,
                               img_root=os.path.realpath('./data'), img_size=args.img_size)
    dataloader = DataLoader(
        wlfwdataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    wlfw_val_dataset = PFLDDatasets(args.val_dataroot, transform,
                                    img_root=os.path.realpath('./data'), img_size=args.img_size)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    weighted_losses = []
    train_losses = []
    val_losses = []
    val_nme = 1e6
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, plfd_backbone, auxiliarynet,
                                                criterion, optimizer, epoch)

        if epoch % args.epoch_interval == 0:
            filename = os.path.join(str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth')
            save_checkpoint({
                'epoch': epoch,
                'plfd_backbone': plfd_backbone.state_dict(),
                'auxiliarynet': auxiliarynet.state_dict()
            }, filename)

        val_loss, cur_val_nme = validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion)
        if cur_val_nme < val_nme:
            filename = os.path.join(str(args.snapshot), "checkpoint_min_nme.pth")
            save_checkpoint({
                'epoch': epoch,
                'plfd_backbone': plfd_backbone.state_dict(),
                'auxiliarynet': auxiliarynet.state_dict()
            }, filename)
            val_nme = cur_val_nme
        scheduler.step(val_loss)

        weighted_losses.append(weighted_train_loss.item())
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        logging.info("epoch: {}, weighted_train_loss: {:.4f}, trainset loss: {:.4f}  valset loss: {:.4f}  best val "
                     "nme: {:.4f}\n ".format(epoch, weighted_train_loss, train_loss, val_loss, val_nme))

    weighted_losses = " ".join(list(map(str, weighted_losses)))
    train_losses = " ".join(list(map(str, train_losses)))
    val_losses = " ".join(list(map(str, val_losses)))
    logging.info(weighted_losses)
    logging.info(train_losses)
    logging.info(val_losses)


def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('-j', '--workers', default=16, type=int)
    parser.add_argument('--backbone', default='v2', type=str, choices=["v2", "v3", "ghost", "lite"])
    parser.add_argument('--devices_id', default='0', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)

    # training
    # -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)
    parser.add_argument('--img_size', default=112, type=int)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=200, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoint/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoint/train.log", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoint/tensorboard", type=str)
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH')  # TBD
    parser.add_argument('--epoch_interval', default=1, type=int)

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./data/train_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='./data/test_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    args.snapshot = os.path.join(args.snapshot, args.backbone)
    args.log_file = os.path.join(args.snapshot, 'train_{}.log'.format(args.backbone))
    os.makedirs(args.snapshot, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
