from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
import math
import numpy as np

from data import WiderFaceDetection, detection_collate, preproc
from config import cfg_re50
from loss.retinaface_loss import RetinaFaceLoss
from loss.prior_box import PriorBox
from models.retinaface import RetinaFace


parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--save_folder', default='./checkpoints/', help='Location to save checkpoint models and params')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = cfg_re50

rgb_mean = (104, 117, 123)              #order blue green red 
num_classes = 2                         #[0 non-face,1-face]
img_dim = cfg['image_size']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = 0
num_gpu = cfg['ngpu']
momentum = 0.9                          #momentum for SGD
weight_decay = 5e-4                     #Weight decay for SGD
initial_lr = 1e-3
gamma = 0.1                             # for LR adjustment
training_dataset = args.training_dataset
save_folder = args.save_folder

resume_net = '/app/Detection/checkpoints/Resnet50_epoch_33.pth'
resume_epoch = 33

model = RetinaFace(num_classes)

if resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    model = torch.nn.DataParallel(model).cuda()
else: 
    model = model.cuda()

cudnn.benchmark = True


optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
loss_function = RetinaFaceLoss(num_classes,0.5,True, 0, True, 5, 0.5, False)

priorbox = PriorBox(image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    model.train()
    epoch = 0 + resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))
    dl_train=data.DataLoader(dataset, batch_size, shuffle=True, 
                                                num_workers=num_workers, collate_fn=detection_collate)

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if resume_epoch > 0:
        start_iter = resume_epoch * epoch_size
    else:
        start_iter = 0

    best_loss = 100000
    
    for iteration in range(start_iter, max_iter):
        loss_batch = []
        if iteration % epoch_size == 0:
            batch_iterator = iter(dl_train)
            #if epoch > 1:
            #    if np.mean(loss_batch) < best_loss:
            #        best_loss = np.mean(loss_batch)
            #        torch.save(model.state_dict(), save_folder + cfg['name']+ 'best_epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        embeddings = model(images)

        optimizer.zero_grad()
        loss = loss_function(embeddings, priors, targets)
        loss_batch.append(loss)
        print(float(loss))
        loss.backward()
        optimizer.step()

        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loss: {:.4f} || LR: {:.4f}'
                .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                epoch_size, iteration + 1, max_iter, loss, lr))

    torch.save(model.state_dict(), save_folder + cfg['name'] + '_Final.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
