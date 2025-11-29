# training.py
""" module for training new deepCR-mask models
"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from tqdm import tqdm_notebook as tqdm_notebook
from astropy.visualization import ZScaleInterval, ImageNormalize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from util import maskMetric
from dataset import dataset, DatasetSim, PairedDatasetImagePath
from unet import WrappedModel, UNet2Sigmoid, UNet2SigmoidVI

__all__ = 'train'


class VoidLRScheduler:
    def __init__(self):
        pass

    def _reset(self):
        pass

    def step(self):
        pass


class train:

    def __init__(self, image, mask=None, ignore=None, sky=None, mode='pair',
                 aug_sky=(0, 0), aug_img=(1, 1), noise=False, saturation=1e5,
                 n_mask_train=1, n_mask_val=1, norm=False, percentile_limit=50,
                 name='model', hidden=32, epoch=50,
                 epoch_phase0=None, batch_size=16, lr=0.005,
                 auto_lr_decay=True, lr_decay_patience=4,
                 lr_decay_factor=0.1, save_after=1e5, plot_every=10,
                 verbose=True, use_tqdm=False,
                 use_tqdm_notebook=False, directory='./',
                 bayesian=False, kl_beta=1e-6):

        """
        This is the class for training deepCR-mask.
        bayesian: if True, use UNet2SigmoidVI + ELBO (BCE + KL)
        kl_beta: scaling factor for KL term in ELBO.
        """
        if torch.cuda.is_available():
            gpu = True
        else:
            gpu = False
            print('No GPU detected on this device! Training on CPU.')

        if mode == 'pair':
            if type(image[0]) == str or type(image[0]) == np.str_:
                data_train = PairedDatasetImagePath(image, aug_sky[0], aug_sky[1], part='train')
                data_val = PairedDatasetImagePath(image, aug_sky[0], aug_sky[1], part='val')
            else:
                data_train = dataset(image, mask, ignore, sky, part='train', aug_sky=aug_sky)
                data_val = dataset(image, mask, ignore, sky, part='val', aug_sky=aug_sky)
        elif mode == 'simulate':
            data_train = DatasetSim(image, mask, sky, aug_sky=aug_sky, aug_img=aug_img, saturation=saturation,
                                    norm=norm, percentile_limit=percentile_limit, part='train',
                                    noise=noise, n_mask=n_mask_train)
            data_val = DatasetSim(image, mask, sky, aug_sky=aug_sky, aug_img=aug_img, saturation=saturation,
                                  norm=norm, percentile_limit=percentile_limit, part='val',
                                  noise=noise, n_mask=n_mask_val)
        else:
            raise TypeError('Mode must be one of pair or simulate!')

        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=8)
        self.shape = data_train[0][0].shape[1]
        self.name = name

        self.bayesian = bayesian
        self.kl_beta = kl_beta
        self.N_train = len(data_train)

        if gpu:
            self.dtype = torch.cuda.FloatTensor
            self.dint = torch.cuda.ByteTensor
            if bayesian:
                self.network = nn.DataParallel(UNet2SigmoidVI(1, 1, hidden))
            else:
                self.network = nn.DataParallel(UNet2Sigmoid(1, 1, hidden))
            self.network.type(self.dtype)
        else:
            self.dtype = torch.FloatTensor
            self.dint = torch.ByteTensor
            if bayesian:
                self.network = WrappedModel(UNet2SigmoidVI(1, 1, hidden))
            else:
                self.network = WrappedModel(UNet2Sigmoid(1, 1, hidden))
            self.network.type(self.dtype)

        # convenience handle to underlying net for KL
        if isinstance(self.network, nn.DataParallel):
            self.base_net = self.network.module
        else:
            self.base_net = self.network

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        if auto_lr_decay:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=lr_decay_factor,
                                                  patience=lr_decay_patience,
                                                  cooldown=2, verbose=True,
                                                  threshold=0.005)
        else:
            self.lr_scheduler = VoidLRScheduler()
        self.lr = lr
        self.BCELoss = nn.BCELoss()
        self.validation_loss = []
        self.epoch_mask = 0
        self.save_after = save_after
        self.n_epochs = epoch
        if epoch_phase0 is None:
            self.n_epochs_phase0 = int(self.n_epochs * 0.4 + 0.5)
        else:
            self.n_epochs_phase0 = epoch_phase0
        self.every = plot_every
        self.directory = directory
        self.verbose = verbose
        self.mode0_complete = False

        if use_tqdm_notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
        self.disable_tqdm = not (use_tqdm_notebook or use_tqdm)

        self.writer = SummaryWriter(log_dir=directory)

    def set_input(self, img0, mask, ignore):
        img0 = (img0 - img0.mean())/img0.std()
        self.img0 = Variable(img0.type(self.dtype)).view(-1,1, self.shape, self.shape)
        self.mask = Variable(mask.type(self.dtype)).view(-1,1, self.shape, self.shape)
        self.ignore = Variable(ignore.type(self.dtype)).view(-1,1, self.shape, self.shape)

    def validate_mask(self, epoch=None):
        torch.random.manual_seed(0)
        np.random.seed(0)
        lmask = 0; count = 0
        metric = np.zeros(4)
        for i, dat in enumerate(self.ValLoader):
            n = dat[0].shape[0]
            count += n
            self.set_input(*dat)
            self.pdt_mask = self.network(self.img0)
            loss = self.backward_network()
            lmask += float(loss.detach()) * n
            metric += maskMetric(self.pdt_mask.reshape(-1, self.shape, self.shape).detach().cpu().numpy() > 0.5,
                                 dat[1].numpy())
        lmask /= count
        TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if self.verbose:
            print('[TPR=%.3f, FPR=%.3f] @threshold = 0.5' % (TPR, FPR))
        if epoch:
            self.writer.add_scalar('TPR', TPR, epoch)
            self.writer.add_scalar('FPR', FPR, epoch)
            self.writer.add_scalar('validate_loss', lmask, epoch)

        return (lmask)

    def train(self):
        if self.verbose:
            print('Begin first {} epochs of training'.format(self.n_epochs_phase0))
            print('Use batch statistics for batch normalization; keep running statistics to be used in phase1')
            print('')
        self.train_phase0(self.n_epochs_phase0)

        filename = self.save()
        self.load(filename)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr/2.5)

        if self.verbose:
            print('Continue onto next {} epochs of training'.format(self.n_epochs - self.n_epochs_phase0))
            print('Batch normalization running statistics frozen and used')
            print('')
        self.train_phase1(self.n_epochs - self.n_epochs_phase0)

    def train_phase0(self, epochs):
        self.network.train()
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):
            for t, dat in enumerate(self.TrainLoader):
                self.optimize_network(dat)
            self.epoch_mask += 1

            if self.epoch_mask % self.every == 0:
                self.plot_example()

            if self.verbose:
                print('----------- epoch = %d -----------' % self.epoch_mask)
            val_loss = self.validate_mask(epoch)
            self.validation_loss.append(val_loss)
            if self.verbose:
                print('loss = %.4f' % (self.validation_loss[-1]))
            if (np.array(self.validation_loss)[-1] == np.array(
                    self.validation_loss).min() and self.epoch_mask > self.save_after):
                filename = self.save()
                if self.verbose:
                    print('Saved to {}.pth'.format(filename))
            self.lr_scheduler.step(self.validation_loss[-1])
            if self.verbose:
                print('')

    def train_phase1(self, epochs):
        self.set_to_eval()
        self.lr_scheduler._reset()
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):
            for t, dat in enumerate(self.TrainLoader):
                self.optimize_network(dat)
            self.epoch_mask += 1

            if self.epoch_mask % self.every==0:
                self.plot_example()

            if self.verbose:
                print('----------- epoch = %d -----------' % self.epoch_mask)
            valLossMask = self.validate_mask(epoch)
            self.validation_loss.append(valLossMask)
            if self.verbose:
                print('loss = %.4f' % (self.validation_loss[-1]))
            if (np.array(self.validation_loss)[-1] == np.array(
                    self.validation_loss).min() and self.epoch_mask > self.save_after):
                filename = self.save()
                if self.verbose:
                    print('Saved to {}.pth'.format(filename))
            self.lr_scheduler.step(self.validation_loss[-1])
            if self.verbose:
                print('')

    def plot_example(self):
        plt.figure(figsize=(30,10))
        plt.subplot(131)
        plt.imshow(np.log(self.img0[0, 0].detach().cpu().numpy()), cmap='gray')
        plt.title('epoch=%d' % self.epoch_mask)
        plt.subplot(132)
        plt.imshow(self.pdt_mask[0, 0].detach().cpu().numpy() > 0.5, cmap='gray')
        plt.title('prediction > 0.5')
        plt.subplot(133)
        plt.imshow(self.mask[0, 0].detach().cpu().numpy(), cmap='gray')
        plt.title('ground truth')
        filename = 'epoch%d' % self.epoch_mask
        print('Save trainplot')
        plt.savefig(self.directory+self.name+filename+'trainplot.png')

    def set_to_eval(self):
        self.network.eval()

    def optimize_network(self, dat):
        self.set_input(*dat)
        self.pdt_mask = self.network(self.img0)
        self.optimizer.zero_grad()
        loss = self.backward_network()
        loss.backward()
        self.optimizer.step()

    def backward_network(self):
        data_loss = self.BCELoss(self.pdt_mask * (1 - self.ignore), self.mask * (1 - self.ignore))
        if self.bayesian:
            kl = self.base_net.kl_loss()
            kl_scaled = kl / float(self.N_train)
            loss = data_loss + self.kl_beta * kl_scaled
        else:
            loss = data_loss
        return loss

    def plot_loss(self):
        plt.figure(figsize=(10,5))
        plt.plot(range(self.epoch_mask), self.validation_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Validation loss')
        plt.savefig(self.directory+self.name+'trainplot.png')

    def save(self):
        time = datetime.datetime.now()
        time = str(time)[:10]
        filename = '%s_%s_epoch%d' % (time, self.name, self.epoch_mask)
        torch.save(self.network.state_dict(), self.directory + filename + '.pth')
        return filename

    def load(self, filename):
        self.network.load_state_dict(torch.load(self.directory + filename + '.pth'))
        loc = filename.find('epoch') + 5
        self.epoch_mask = int(filename[loc:])
