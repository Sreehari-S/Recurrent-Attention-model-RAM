import warnings

import numpy as np

import pickle
import os
import torch
import shutil
# from tflogger import TFLogger


class Callback(object):
    '''Abstract base class used to build new callbacks.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    '''
    def __init__(self, model):
        self.model = model

    def on_train_beg(self):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_end(self, epoch, batch, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class PlotCbk(Callback):
    def __init__(self, plot_dir, model, num_imgs, plot_freq, use_gpu):
        self.model = model
        self.num_imgs = num_imgs
        self.plot_freq = plot_freq
        self.use_gpu = use_gpu
        self.plot_dir = os.path.join(plot_dir, self.model.name+'/')
        # print (self.plot_dir)
        # print ('###################################')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def on_batch_end(self, epoch, batch_ind, name='',logs={}):
        imgs = logs['x'][:self.num_imgs].squeeze(1)
        locs = [loc[:self.num_imgs] for loc in logs['locs']]
        preds = logs['preds'][:self.num_imgs] 
        Ys = logs['y'][:self.num_imgs]
        if (epoch % self.plot_freq == 0) and (batch_ind == 0):
            if self.use_gpu:
                imgs = imgs.cpu()
                locs = [l.cpu() for l in locs]
                preds = preds.cpu()
                Ys = Ys.cpu()
            imgs = imgs.data.numpy()
            locs = [l.data.numpy() for l in locs]
            preds = preds.data.numpy()
            Ys = Ys.data.numpy()
            pickle.dump(
                imgs, open(
                    self.plot_dir + "{}g_{}.p".format(name, epoch),
                    "wb"
                )
            )
            pickle.dump(
                preds, open(
                    self.plot_dir + "{}preds_{}.p".format(name, epoch),
                    "wb"
                )
            )
            pickle.dump(
                Ys, open(
                    self.plot_dir + "{}Ys_{}.p".format(name, epoch),
                    "wb"
                )
            )
            pickle.dump(
                locs, open(
                    self.plot_dir + "{}l_{}.p".format(name, epoch),
                    "wb"
                )
            )


# class TensorBoard(Callback):
#     def __init__(self, model, log_dir):
#         self.model = model
#         self.logger = TFLogger(log_dir)

#     def to_np(self, x):
#         return x.data.cpu().numpy()

#     def on_epoch_end(self, epoch, logs):
#         for tag in ['loss', 'acc']:
#             self.logger.scalar_summary(tag, logs[tag], epoch)

#         for tag, value in self.model.named_parameters():
#             tag = tag.replace('.', '/')
#             self.logger.histo_summary(tag, self.to_np(value), epoch)
#             self.logger.histo_summary(tag+'/grad', self.to_np(value.grad), epoch)


class ModelCheckpoint(Callback):
    def __init__(self, model, optimizer, ckpt_dir):
        self.model = model
        self.optimizer = optimizer
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        filename = self.model.name + '_ckpt'
        self.ckpt_path = os.path.join(ckpt_dir, filename)
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs={}):
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                 'optim_state_dict': self.optimizer.state_dict()}

        torch.save(state, self.ckpt_path)

        if logs['val_acc'] > self.best_val_acc:
            self.best_val_acc = logs['val_acc']
            shutil.copyfile(self.ckpt_path, self.ckpt_path + '_best')


class LearningRateScheduler(Callback):
    def __init__(self, scheduler, monitor_val):
        self.scheduler = scheduler
        self.monitor_val = monitor_val

    def on_epoch_end(self, epoch, logs):
        self.scheduler.step(logs[self.monitor_val])


class EarlyStopping(Callback):
    '''Stop training when a monitored quantity has stopped improving.
    '''
    def __init__(self, model, monitor='val_loss', patience=0, verbose=0, mode='auto'):
        '''
        @param monitor: str. Quantity to monitor.
        @param patience: number of epochs with no improvement after which training will be stopped.
        @param verbose: verbosity mode, 0 or 1.
        @param mode: one of {auto, min, max}. Decides if the monitored quantity improves. If set to `max`, increase of the quantity indicates improvement, and vice versa. If set to 'auto', behaves like 'max' if `monitor` contains substring 'acc'. Otherwise, behaves like 'min'.
        '''
        super(EarlyStopping, self).__init__(model)
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires {} available!'.format(self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch {}: early stopping'.format(epoch))
                self.model.stop_training = True
            self.wait += 1
