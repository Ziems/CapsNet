"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys
sys.setrecursionlimit(15000)
import math

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import data_loader as custom_dl

BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 5
NUM_ROUTING_ITERATIONS = 3
CHANNELS = 3

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' %kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride, padding):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class CapsNet(nn.Module):
    def __init__(self, routing_iterations, n_classes=10):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=9, stride=2,padding=0)  # outputs 6*6
        self.num_primaryCaps = 32 * 8 * 8
        routing_module1 = AgreementRouting(self.num_primaryCaps, 64, routing_iterations)
        self.cap1 = CapsLayer(self.num_primaryCaps, 8, 64, 16, routing_module1)
        routing_module2 = AgreementRouting(64, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(64, 16, n_classes, 16, routing_module2)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.cap1(x)
        x = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, caps_net, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.caps_net = caps_net
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.caps_net(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchvision import transforms
    from torchvision.utils import make_grid
    from torchvision.datasets.cifar import CIFAR10
    from logger import Logger

    #root = '/ProxylessGAN'
    root = '.'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu=2
    #device = torch.device("cpu")
    print("using " + str(device))

    print("n_gpu: ", ngpu)

    model = CapsNet(routing_iterations=3, n_classes=10).to(device)
    model.train()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    logger = Logger(root+'/logs/junk/')

    optimizer = Adam(model.parameters(), lr=5e-4)
    capsule_loss = MarginLoss(0.9, 0.1, 0.5)

    train_loader, valid_loader = custom_dl.get_train_valid_loader(data_dir=root+'/data/cifar10/',
                                                              batch_size=BATCH_SIZE,
                                                              augment=False,
                                                              random_seed=1,
                                                              valid_size=0.1)
    test_loader = custom_dl.get_test_loader(data_dir=root+'/data/cifar10/', batch_size=BATCH_SIZE)

    clock = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # one_hot_labels = one_hot_embedding(labels, NUM_CLASSES).to(device)

            optimizer.zero_grad()

            x, class_probs = model(inputs)
            loss = capsule_loss(class_probs, labels)

            loss.backward()
            optimizer.step()

            if clock % 100 == 0:
                _, argmax = torch.max(class_probs, 1)
                labels = labels.cpu()
                argmax = argmax.cpu()
                inputs = inputs.cpu()
                accuracy = (labels == argmax.squeeze()).float().mean()
                print("~[e%d]batch %d~ Loss: %.3f, Train Acc: %.2f"%(epoch, i, loss.item(), accuracy.item()))
                # 1. Log scalar values (scalar summary)
                info = {'loss': loss.item(), 'train accuracy': accuracy.item()}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, clock + 1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), clock+1)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), clock+1)

                # 3. Log training images (image summary)
                info = {'images': inputs.view(-1, 3, 32, 32)[:10, :, :, :].cpu().numpy()}

                for tag, images in info.items():
                    logger.image_summary(tag, images, clock+1)

            clock += 1
        
        # Check validation accuracy after each epoch
        # model.eval()
        # data = next(iter(valid_loader))
        # inputs, labels = data
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        # classes, reconstructions = model(inputs)
        # loss = capsule_loss(inputs, one_hot_labels, classes, reconstructions)
        # _, argmax = torch.max(classes, 1)
        # labels = labels.cpu()
        # argmax = argmax.cpu()
        # inputs = inputs.cpu()
        # accuracy = (labels == argmax.squeeze()).float().mean()
        # info = {'validation accuracy': accuracy.item()}

        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, clock + 1)

    
    print('FINISHED TRAINING')
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            x, class_probs = model(images)
            _, predicted = torch.max(class_probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))




