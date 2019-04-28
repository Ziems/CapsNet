"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import data_loader as custom_dl

BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 100
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

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=CHANNELS, out_channels=512, kernel_size=9, stride=1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=9, stride=1, padding=get_same_padding(9))
        self.dropout2 = nn.Dropout2d(0.5)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)

        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 8 * 8, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32 * 32 * CHANNELS),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.dropout2(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstruction_inputs = (x * y[:, :, None]).view(x.size(0), -1)
        reconstructions = self.decoder(reconstruction_inputs)

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


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

    model = CapsuleNet().to(device)
    model.train()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    logger = Logger(root+'/logs/')

    optimizer = Adam(model.parameters(), lr=5e-4)
    capsule_loss = CapsuleLoss()

    train_loader, _ = custom_dl.get_train_valid_loader(data_dir=root+'/data/cifar10/',
                                                              batch_size=BATCH_SIZE,
                                                              augment=False,
                                                              random_seed=1,
                                                              valid_size=0.0)
    test_loader = custom_dl.get_test_loader(data_dir=root+'/data/cifar10/', batch_size=BATCH_SIZE)

    clock = 0

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            one_hot_labels = one_hot_embedding(labels, NUM_CLASSES).to(device)

            optimizer.zero_grad()

            classes, reconstructions = model(inputs, one_hot_labels)
            loss = capsule_loss(inputs, one_hot_labels, classes, reconstructions)

            loss.backward()
            optimizer.step()

            if clock % 100 == 0:
                _, argmax = torch.max(classes, 1)
                labels = labels.cpu()
                argmax = argmax.cpu()
                inputs = inputs.cpu()
                accuracy = (labels == argmax.squeeze()).float().mean()
                print("~[e%d]batch %d~ Loss: %.3f, Acc: %.2f"%(epoch, i, loss.item(), accuracy.item()))
                # 1. Log scalar values (scalar summary)
                info = {'loss': loss.item(), 'accuracy': accuracy.item()}

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
    
    print('FINISHED TRAINING')
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))




