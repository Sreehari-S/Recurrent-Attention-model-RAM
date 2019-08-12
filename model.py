from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
from modules import RAMNet
import torchnet as tnt
nclasses=3
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
confusion_meter.reset()
class RecurrentAttention(nn.Module):
    def __init__(self, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(RecurrentAttention, self).__init__()
        self.std = args.std
        self.use_gpu = args.use_gpu
        self.M = args.M

        self.num_glimpses = args.num_glimpses
        self.ram_net = RAMNet(args)

        self.name = 'ram_{}_{}_{}_{}x{}_{}_{}'.format(args.model, args.rnn_type, args.num_glimpses, args.patch_size, args.patch_size, args.glimpse_scale, args.num_patches)

    def init_loc(self, batch_size):
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        l_t = torch.Tensor(batch_size, 2).uniform_(-1, 1)
        l_t = Variable(l_t).type(dtype)
        return l_t

    def forward(self, x, y, is_training=False):
        """
        @param x: image. (batch, channel, height, width)
        @param y: word indices. (batch, seq_len)
        """
        if self.use_gpu:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x, volatile=not is_training), Variable(y, volatile=not is_training)

        if not is_training:
            with torch.no_grad():
                return self.forward_test(x, y)

        batch_size = x.shape[0]

        # Main forwarding step
        # locs:             (batch, 2)*num_glimpses
        # baselines:        (batch, num_glimpses)
        # log_pi:           (batch, num_glimpses)
        # log_probas:       (batch, num_class)
        l_t = self.init_loc(batch_size)
        locs, baselines, log_pi, log_probas = self.ram_net(x, l_t)

        # Prediction Loss & Reward
        # preds:    (batch)
        # reward:           (batch)
        preds = torch.max(log_probas, 1)[1]
        reward = (preds.detach() == y).float()
        loss_pred = F.nll_loss(log_probas, y)
        # print(x.size(), preds, y, locs)
        # assert False
        # Baseline Loss
        # reward:          (batch, num_glimpses)
        reward = reward.unsqueeze(1).repeat(1, self.num_glimpses)
        loss_baseline = F.mse_loss(baselines, reward)

        # Reinforce Loss
        adjusted_reward = reward - baselines.detach()
        # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
        loss_reinforce = torch.mean(-log_pi*adjusted_reward)
        # loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
        # loss_reinforce = torch.mean(loss_reinforce)

        # sum up into a hybrid loss
        loss = loss_pred + loss_baseline + loss_reinforce

        # calculate accuracy
        # preds:    (batch)
        # correct:  (batch)
        correct = (preds == y).float()
        acc = 100 * (correct.sum() / len(y))

        return {'loss': loss,
                'acc': acc,
                'locs': locs,
                'x': x,
                'preds': preds,
                'y': y}

    def forward_test(self, x, y):
        # duplicate 10 times
        x = x.repeat(self.M, 1, 1, 1)

        batch_size = x.shape[0]
        # Main forwarding step
        # locs:             (batch*M, 2)*num_glimpses
        # baselines:        (batch*M, num_glimpses)
        # log_pi:           (batch*M, num_glimpses)
        # log_probas:       (batch*M, num_class)
        l_t = self.init_loc(batch_size)
        locs, baselines, log_pi, log_probas = self.ram_net(x, l_t)

        # Average
        log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
        log_probas = torch.mean(log_probas, dim=0)

        baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
        baselines = torch.mean(baselines, dim=0)

        log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
        log_pi = torch.mean(log_pi, dim=0)

        # Prediction Loss & Reward
        # preds:    (batch)
        # reward:           (batch)
        preds = torch.max(log_probas, 1)[1]
        reward = (preds.detach() == y).float()
        loss_pred = F.nll_loss(log_probas, y)
        confusion_meter.add(preds.data.view(-1),y.data.view(-1))

        # Baseline Loss
        # reward:          (batch, num_glimpses)
        reward = reward.unsqueeze(1).repeat(1, self.num_glimpses)
        loss_baseline = F.mse_loss(baselines, reward)

        # Reinforce Loss
        adjusted_reward = reward - baselines.detach()
        # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
        loss_reinforce = torch.mean(-log_pi*adjusted_reward)
        # loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
        # loss_reinforce = torch.mean(loss_reinforce)

        # sum up into a hybrid loss
        loss = loss_pred + loss_baseline + loss_reinforce

        # calculate accuracy
        # preds:    (batch)
        # correct:  (batch)
        correct = (preds == y).float()
        acc = 100 * (correct.sum() / len(y))

        return {'loss': loss,
                'acc': acc,
                'con_mat':confusion_meter}
        # return {'loss': loss,
        #         'acc': acc}