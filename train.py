import argparse
import logging
from pathlib import Path
import os

import numpy as np
import torch

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from dataset.datasets import WLFWDatasets
from models.ghost_pfld import PFLDInference, AuxiliaryNet
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


def validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion):
    plfd_backbone.eval()
    auxiliarynet.eval() 
    losses = []
    with torch.no_grad():
        for img, landmark_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = plfd_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(args.log_file, mode='w'),
                                  logging.StreamHandler()
                                  ]
                        )
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    plfd_backbone = PFLDInference().to(device)
    auxiliarynet = AuxiliaryNet().to(device)
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
    wlfwdataset = WLFWDatasets(args.dataroot, transform, img_root=os.path.realpath('./data'))
    dataloader = DataLoader(
        wlfwdataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transform, img_root=os.path.realpath('./data'))
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    weighted_losses = []
    train_losses = []
    val_losses = []
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

        val_loss = validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion)

        scheduler.step(val_loss)

        weighted_losses.append(weighted_train_loss.item())
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        logging.info("epoch: {}, weighted_train_loss: {:.4f}, train loss: {:.4f}  val:loss: {:.4f}\n"
                     .format(epoch, weighted_train_loss, train_loss, val_loss))

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
    parser.add_argument('--devices_id', default='0', type=str)  # TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    # -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=1000, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoint/snapshot/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoint/train.logs", type=str)
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
    parser.add_argument('--train_batchsize', default=8, type=int)
    parser.add_argument('--val_batchsize', default=1, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
