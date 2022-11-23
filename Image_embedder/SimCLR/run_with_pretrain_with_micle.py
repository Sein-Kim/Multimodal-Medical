import argparse
from codecs import namereplace_errors
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr_micle import SimCLR_micle
from load_data import load_data
from dataloader import TensorData
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', type='str', default='ADNI')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=1024, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu_index', default=0, type=int, help='Gpu index.')

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(args.gpu_index)

        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    print(args.device)
    print("preparing images")
    train_images, train_labels, extract_images, extract_names = load_data(args)
    print("load images -- end")
    train_data_testing = TensorData(train_images, train_labels)
    #for inductive setting
    train_loader = torch.utils.data.DataLoader(
        train_data_testing, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)	 
    ##########



    #load model
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    with torch.cuda.device(args.gpu_index):
        checkpoint = torch.load('./runs/Pretrain_model_emb1024/checkpoint_0100.pth.tar', map_location=args.device)

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR_micle(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)

        extract_names = np.array(extract_names)
        model.eval()
        kkk = 0
        for im in extract_images:
            if kkk == 0:
                im_tf = im.to(args.device)
                anchor_features = model(im_tf)
                feat = torch.mean(anchor_features, dim=0).unsqueeze(0).detach().cpu()
            else:
                im_tf = im.to(args.device)
                anchor_features = model(im_tf)
                feat_ = torch.mean(anchor_features, dim=0).unsqueeze(0).detach().cpu()
                feat = torch.cat((feat,feat_),dim=0)
                del feat_
            kkk+=1
    n = feat.detach().cpu().numpy()
    np.savetxt('./extracted_feature/train_feature.csv',n,delimiter=',')
    np.savetxt('./extracted_feature/train_id.csv',extract_names,delimiter=',')


if __name__ == "__main__":
    main()
