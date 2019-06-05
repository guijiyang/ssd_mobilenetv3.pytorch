# %%
import torchvision
from torchvision import transforms, datasets
import torchsummary
import logging
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import pandas as pd
os.chdir('/home/guijiyang/Code/python/torch/ssd')
from ssd_utils import Logger
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small

HOME=os.path.expanduser('~')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 200
DATA_DIR = HOME+ '/dataset/cifar10'
OUTPUT_DIR=HOME+'/output'
# print(os.getcwd())
#%%
train_data=datasets.CIFAR10(DATA_DIR,
                     train=False,
                     transform=transforms.Compose([
                         transforms.Resize(224),
#                          transforms.RandomAffine(10),
#                          transforms.RandomRotation(10),
#                          transforms.RandomHorizontalFlip(0.5),
#                          transforms.RandomVerticalFlip(0.5),
                         transforms.ToTensor(),
#                          transforms.Normalize((0,0,0),(255.,255.,255.))
                     ]))

idx=np.random.randint(len(train_data))
print(idx)
img=train_data[idx][0]
print(img.max())
lbl=train_data[idx][1]
# img=img.reshape((1,)+img.shape)
img=img.numpy()

#%%
def train(model, device, train_loader, optimizer, epoch, scheduler, logger):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清除累计梯度
        output = model(data)
        loss = F.cross_entropy(output, target)
        losses.append(loss)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 500 == 0:
            logger('Train Epoch : {} [{}/{}]\t lr: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                scheduler.get_lr()[0], loss.item()))
    avg_loss = sum(losses)/len(losses)
    dict_info = {'epoch': epoch, 'loss': avg_loss.item()}
    if epoch == 1:
        df = pd.DataFrame([dict_info])
        df.to_csv('./output/train_result.csv', index=False, encoding='utf-8')
    else:
        df = pd.DataFrame([dict_info])
        df.to_csv('./output/train_result.csv', header=False,
                  index=False, mode='a+', encoding='utf-8')


def test(model, device, test_loader, epoch, logger):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger('\nTest set:Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    dict_info = {'epoch': epoch, 'eval_loss': test_loss, 'accuracy': accuracy}
    if epoch == 1:
        df = pd.DataFrame([dict_info])
        df.to_csv('./output/eval_result.csv', index=False, encoding='utf-8')
    else:
        df = pd.DataFrame([dict_info])
        df.to_csv('./output/eval_result.csv', header=False,
                  index=False, mode='a+', encoding='utf-8')

model = mobilenetv3_small(num_classes=10).to(DEVICE)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=3, gamma=0.9)
logger = Logger(HOME+'/log', 'mobilenetv3_small')

# %%
train_loader=torch.utils.data.DataLoader(
    datasets.CIFAR10(DATA_DIR,
                     train=True,
                     transform=transforms.Compose([
                         transforms.Resize(224),
                         transforms.RandomAffine(10),
                         transforms.RandomRotation(10),
                         transforms.RandomHorizontalFlip(0.5),
                         transforms.RandomVerticalFlip(0.5),
                         transforms.ToTensor(),
                         transforms.Normalize((0,0,0),(255.,255.,255.))
                     ])), batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(DATA_DIR,
                     train=False,
                     transform=transforms.Compose([
                         transforms.Resize(224),
                         transforms.ToTensor(),
                         transforms.Normalize((0, 0, 0), (255., 255., 255.))
                     ])), batch_size=32, shuffle=True)


#%%
load_weight = torch.load('./pretrained/mobilenetv3-small-c7eb32fe.pth')

train_weights=['classifier.3.weight',
    'classifier.3.bias',
    'classifier.4.weight',
    'classifier.4.bias',
    'classifier.4.running_mean',
    'classifier.4.running_var']

for train_weight in train_weights:
    load_weight.pop(train_weight)

model.load_state_dict(load_weight, strict=False)
#%%
params=model.named_parameters()
for param in params:
    if param[0] not in train_weights:
        param[1].requires_grad=False
        print('{} : False'.format(param[0]))
    else:
        param[1].requires_grad=True
        print('{} : True'.format(param[0]))
#%%
torchsummary.summary(model, (3,224,224))

#%%
# 可视化图片特征图谱随着forward传递的变化
layers=model.get_layers()
print(layers)

# %%
# 训练模型
for epoch in range(1, EPOCHS+1):
    train(model, DEVICE, train_loader, optimizer, epoch, scheduler, logger)
    test(model, DEVICE, test_loader,epoch, logger)
    scheduler.step()
    if epoch%10==0:
        ## save trained parameters
        torch.save(model.state_dict(), OUTPUT_DIR+'/mobilenetv3_small_{}.pth'.format(epoch))


#%%
# model = mobilenetv3_small(num_classes=10, visual=True, vis_layers=[1, 3, 7, 11]).to(DEVICE)
# optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)