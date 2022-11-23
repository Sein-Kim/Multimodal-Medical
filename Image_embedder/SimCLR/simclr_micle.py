import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import numpy as np

from medical_aug import medical_aug
torch.manual_seed(0)


class SimCLR_micle(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.mse = torch.nn.MSELoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
    
    def micle(self, features, indx):
        index = indx.detach().numpy()
        unique, counts = np.unique(index, return_counts=True)
        count_dict = dict(zip(unique, counts))
        loss = torch.zeros(1).to(self.args.device)
        count = 0
        len_dict = 0
        for key in count_dict:
            len_dict +=1
            if count_dict[key]>2:
                which = np.where(index == key)[0]
                mask = torch.tensor(which).to(self.args.device)

                features_ = features[mask]
                features_ = F.normalize(features_, dim=1)

                similarity_matrix = torch.matmul(features_, features_.T)
                similarity_matrix = F.normalize(similarity_matrix)
                mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(self.args.device)
                positive = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

                labels = torch.ones(positive.shape,dtype=torch.long).to(self.args.device)

                loss += self.mse(positive.to(torch.float32),labels.to(torch.float32))
                count +=1
        if not (count==0):
            loss =loss/count
        return loss

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)


        n_iter = 0

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                for i in range(images.shape[0]):
                    aug = medical_aug(images[i])
                    images = torch.cat((images,aug.unsqueeze(0)),0)
                    _ = torch.cat((_,_[i].unsqueeze(1)),0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    micle_loss = self.micle(features, _)

                self.optimizer.zero_grad()
                overall_loss = loss + micle_loss

                
                overall_loss = overall_loss.to(torch.float32)
                scaler.scale(overall_loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()

        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename=checkpoint_name)
