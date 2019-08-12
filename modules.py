import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.autograd import Variable
import torchvision.models as models
import numpy as np


class retina(object):
    def __init__(self, patch_size, num_patches, scale, use_gpu):
        """
        @param patch_size: side length of the extracted patched.
        @param num_patches: number of patches to extract in the glimpse.
        @param scale: scaling factor that controls the size of successive patches.
        """
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.scale = scale
        self.use_gpu = use_gpu

    def foveate(self, x, l, flatten=True):
        """
        Extract `num_patches` square patches,  centered at location `l`.
        The initial patch is a square of sidelength `patch_size`,
        and each subsequent patch is a square whose sidelength is `scale`
        times the size of the previous patch.  All patches are finally
        resized to the same size of the first patch and then flattened.

        @param x: img. (batch, height, width, channel)
        @param l: location. (batch,2)
        @return Variable: (batch, num_patches*channel*patch_size*patch_size).
        """
        patches = []
        size = self.patch_size
        # extract num_patches patches of increasing size
        for i in range(self.num_patches):
            patches.append(self.extract_patch(x, l, size))
            size = int(self.scale * size)

        # resize the patches to squares of size patch_size
        for i in range(1, len(patches)):
            num_patches = patches[i].shape[-1] // self.patch_size
            patches[i] = F.avg_pool2d(patches[i], num_patches)

        # concatenate into a single tensor and flatten
        patches = torch.cat(patches, 1)

        if flatten:
            patches = patches.view(patches.shape[0], -1)

        return patches

    def extract_patch(self, x, l, size):
        """
        @param x: img. (batch, height, width, channel)
        @param l: location. (batch, 2)
        @param size: the size of the extracted patch.
        @return Variable (batch, height, width, channel)
        """
        B, C, H, W = x.shape

        if not hasattr(self, 'imgShape'):
            self.imgShape = torch.FloatTensor([H, W]).unsqueeze(0)
            if self.use_gpu:
                self.imgShape = self.imgShape.cuda()

        # coordins from [-1,1] to H,W scale
        coords = (0.5 * ((l.data + 1.0) * self.imgShape)).long()

        # pad the image with enough 0s
        x = nn.ConstantPad2d(size//2, 0.)(x)

        # calculate coordinate for each batch samle (padding considered)
        from_x, from_y = coords[:, 0], coords[:, 1]
        to_x, to_y = from_x + size, from_y + size
        # The above is the original implementation
        # It only works if the input image is a square
        # The following is the correct implementation
        # from_y, from_x = coords[:, 0], coords[:, 1]
        # to_y, to_x = from_y + size, from_x + size

        # extract the patches
        patch = []
        for i in range(B):
            patch.append(x[i, :, from_y[i]:to_y[i], from_x[i]:to_x[i]].unsqueeze(0))

        return torch.cat(patch)

class ConvNet(nn.Module):
    def __init__(self, num_channel, hidden_g):
        super(ConvNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.conv1 = nn.Conv2d(num_channel, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bnfc = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64, hidden_g)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(self.bn1(x)), 2))
        x = self.drop(x)
        x = F.relu(F.max_pool2d(self.conv2(self.bn2(x)), 2))
        x = x.view(-1, 64)
        x = self.drop(x)
        x = self.fc1(self.bnfc(x))
        return x

class PreTrainedNet(nn.Module):
    def __init__(self, num_channel, hidden_g, net='resnet18'):
        """
num channel ==  equivalent to num of scales times num_channels 
hidden_g    ==  number neurons in the mlp    
network     ==  type of pre-trained network to be used
"""
        super(PreTrainedNet, self).__init__()
        data = Variable(torch.rand(2,num_channel,32,32))
        self.conv1 = nn.Conv2d(num_channel, 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channel)
        if 'resnet' in net:
            model = getattr(models, net)(pretrained=True)
            self.feature = nn.Sequential(*list(model.children())[:-3])           
        if 'densenet' in net:
            model = getattr(models, net)(pretrained=True)
            self.feature = nn.Sequential(*list(model.features.children())[:-3])              
        
        self.pooler = nn.AdaptiveAvgPool2d((1,1))
        out = self.pooler(self.feature(self.conv1(data)))
        shape = out.size()[1]
        self.fc = nn.Linear(shape, hidden_g)
        self.bnfc = nn.BatchNorm1d(shape)
    def forward(self, x):
        x = F.relu(self.conv1(self.bn1(x)))
        x = self.feature(x)
        x = self.pooler(x)
        shape = x.size()[1]
        x = x.view(-1, shape)
        x = self.fc(self.bnfc(x))
        return x

class GlimpseNet(nn.Module):
    def __init__(self, hidden_g, hidden_l, patch_size, num_patches, scale, num_channel, use_gpu, conv, model):
        """
        @param hidden_g: hidden layer size of the fc layer for `phi`.
        @param hidden_l: hidden layer size of the fc layer for `l`.
        @param patch_size: size of the square patches in the glimpses extracted
        @param by the retina.
        @param num_patches: number of patches to extract per glimpse.
        @param scale: scaling factor that controls the size of successive patches.
        @param num_channel: number of channels in each image.
        """
        super(GlimpseNet, self).__init__()
        self.retina = retina(patch_size, num_patches, scale, use_gpu)
        
        self.flatten = not conv
        # glimpse layer
        if conv:
            if model == 'vanilla':
                self.fc1 = ConvNet(num_channel*num_patches, hidden_g)
            else:
                self.fc1 = PreTrainedNet(num_channel*num_patches, hidden_g, net=model) 
        else:
            D_in = num_patches*patch_size*patch_size*num_channel
            self.fc1 = nn.Linear(D_in, hidden_g)

        # location layer
        self.fc2 = nn.Linear(2, hidden_l)

        self.fc3 = nn.Linear(hidden_g, hidden_g+hidden_l)
        self.fc4 = nn.Linear(hidden_l, hidden_g+hidden_l)

    def forward(self, x_t, l_t):
        """
        Combines the "what" and the "where" into a glimpse feature vector. Extract `num_patches` different resolution patches of the same size (patch_size, patch_size) to get "what". Then combine it with a two dimension "where" vector, each of which element is ranging in [-1,1].

        @param x_t: (batch, height, width, channel)
        @param l_t: (batch, 2)
        @return output: (batch, hidden_g+hidden_l)
        """
        glimpse = self.retina.foveate(x_t, l_t, flatten=self.flatten)

        what = self.fc3(F.relu(self.fc1(glimpse)))
        where = self.fc4(F.relu(self.fc2(l_t)))

        g = F.relu(what + where)

        return g


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - rnn_hidden: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, rnn_hidden). The glimpse
      representation returned by the glimpse network for the current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, rnn_hidden). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, rnn_hidden). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, rnn_hidden, use_gpu, rnn_type='RNN'):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.rnn_hidden = rnn_hidden
        self.use_gpu = use_gpu
        self.rnn_type = rnn_type
        if rnn_type=='RNN':
            self.rnn = nn.RNNCell(input_size, rnn_hidden, bias=True, nonlinearity='relu')
        if rnn_type=='LSTM':
            self.rnn = nn.LSTMCell(input_size, rnn_hidden, bias=True)
    
    def forward(self, g_t, h_t_prev):
        h_t = self.rnn(g_t, h_t_prev)
        return h_t

    def init_hidden(self, batch_size, use_gpu=False):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        if self.rnn_type == 'RNN':
            h_t = torch.zeros(batch_size, self.rnn_hidden)
            h_t = Variable(h_t).type(dtype)
        elif self.rnn_type == 'LSTM':
            h = torch.zeros(batch_size, self.rnn_hidden)
            h = Variable(h).type(dtype)
            c = torch.zeros(batch_size, self.rnn_hidden)
            c = Variable(c).type(dtype)
            h_t = (h, c)
        return h_t


class ActionNet(nn.Module):
    def __init__(self, input_size, output_size):
        """
        @param input_size: input size of the fc layer.
        @param output_size: output size of the fc layer.
        """
        super(ActionNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        """
        Uses the last output of the decoder to output the final classification.
        @param h_t: (batch, rnn_hidden)
        @return a_t: (batch, output_size)
        """
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class LocationNet(nn.Module):
    def __init__(self, input_size, output_size, std):
        """
        @param input_size: input size of the fc layer.
        @param output_size: output size of the fc layer.
        @param std: standard deviation of the normal distribution.
        """
        super(LocationNet, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        """
        Generate next location `l_t` by calculating the coordinates
        conditioned on an affine and adding a normal noise followed
        by a tanh to clamp the output beween [-1, 1].
        @param h_t: hidden state. (batch, rnn_hidden)
        @return mu: noise free location. Used for calculating
                    reinforce loss. (B, 2).
        @return l_t: Next location. (B, 2).
        """
        # compute noise-free location
        mu = F.tanh(self.fc(h_t))

        # sample from gaussian parametrized by this mean
        # This is the origin repo implementation
        noise = torch.from_numpy(np.random.normal(
            scale=self.std, size=mu.shape)
        )
        noise = Variable(noise.float()).type_as(mu)

        # This is an equivalent implementation
        # noise = torch.zeros_like(mu)
        # noise.data.normal_(std=self.std)

        # add noise to the location and bound between [-1, 1]
        l_t = mu + noise
        l_t = F.tanh(l_t)

        # prevent gradient flow
        # Note that l_t is not used to calculate gradients later.
        # Hence detach it explicitly here.
        l_t = l_t.detach()

        return mu, l_t


class BaselineNet(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(BaselineNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t))
        return b_t


class RAMNet(nn.Module):
    def __init__(self, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(RAMNet, self).__init__()
        rnn_inp_size = args.glimpse_hidden + args.loc_hidden
        self.num_glimpses = args.num_glimpses
        self.std = args.std

        self.glimpse_net = GlimpseNet(args.glimpse_hidden, args.loc_hidden, args.patch_size, args.num_patches, args.glimpse_scale, args.num_channels, args.use_gpu, args.conv, args.model)
        self.rnn = core_network(rnn_inp_size, args.rnn_hidden, args.use_gpu, args.rnn_type)
        self.location_net = LocationNet(args.rnn_hidden, 2, args.std)
        self.classifier = ActionNet(args.rnn_hidden, args.num_class)
        self.baseline_net = BaselineNet(args.rnn_hidden, 1)

    def step(self, x, l_t, h_t):
        """
        @param x: image. (batch, channel, height, width)
        @param l_t: location trial. (batch, 2)
        @param h_t: last hidden state. (batch, rnn_hidden)
        @return h_t: next hidden state. (batch, rnn_hidden)
        @return l_t: next location trial. (batch, 2)
        @return b_t: baseline for step t. (batch)
        @return log_pi: probability for next location trial. (batch)
        """
        glimpse = self.glimpse_net(x, l_t)
        h_t = self.rnn(glimpse, h_t)
        mu, l_t = self.location_net(h_t)
        b_t = self.baseline_net(h_t).squeeze()

        log_pi = Normal(mu, self.std).log_prob(l_t)
        # Note: log(p_y*p_x) = log(p_y) + log(p_x)
        log_pi = log_pi.sum(dim=1)

        return h_t, l_t, b_t, log_pi

    def forward(self, x, l_t):
        """
        @param x: image. (batch, channel, height, width)
        @param l_t: initial location. (batch, 2)

        @return hiddens: hidden states (output) of rnn. (batch, num_glimpses, rnn_hidden)
        @return locs: locations. (batch, 2)*num_glimpses
        @return baselines: (batch, num_glimpses)
        @return log_pi: probabilities for each location trial. (batch, num_glimpses)
        """
        batch_size = x.shape[0]
        h_t = self.rnn.init_hidden(batch_size)

        locs = []
        baselines = []
        log_pi = []
        for t in range(self.num_glimpses):
            h_t, l_t, b_t, p_t = self.step(x, l_t, h_t)
            locs.append(l_t)
            baselines.append(b_t)
            log_pi.append(p_t)

        log_probas = self.classifier(h_t)
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)
        return locs, baselines, log_pi, log_probas
