import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os

import torchvision
from models.resnet import ResNet18,ResNet50,ResNetT
from models.mobilenetv2 import MobileNetV2
from models.senet import SENet18
import argparse

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 训练函数
def train(rank, world_size,model):
    setup(rank, world_size)
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # 定义模型和优化器
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # 训练循环
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    cleanup()

def loadDataAndModel(args):
    print('==> Preparing data..' + args.dataset)
    if args.dataset == 'tiny':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_RandAug = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AugMix = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            #transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(64/0.875)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if True:
            if args.trainaug == 0:
                transform_train = transform_train
            elif args.trainaug == 3:
                transform_train = transform_train_AugMix
            else:
                transform_train = transform_train_RandAug
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AutoAug = transforms.Compose([
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform_train_RandAug = transforms.Compose([
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AugMix = transforms.Compose([
            #transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if True:
            if args.trainaug == 0:
                transform_train = transform_train
            elif args.trainaug == 1:
                transform_train = transform_train_AutoAug
            elif args.trainaug == 2:
                transform_train = transform_train_RandAug
            elif args.trainaug == 3:
                transform_train = transform_train_AugMix
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    # Dataset
    print('==> Building model..')
    if args.dataset == 'cifar10': #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        cls_outdim = 10
        trainset = torchvision.datasets.CIFAR10(root='/home/caifei/Project/Datasets/cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/home/caifei/Project/Datasets/cifar10', train=False, download=True, transform=transform_test)
        wholedataset = torchvision.datasets.CIFAR10(root='/home/caifei/Project/Datasets/cifar10', train=True, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        cls_outdim = 100
        trainset = torchvision.datasets.CIFAR100(root='/home/caifei/Project/Datasets/cifar100', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='/home/caifei/Project/Datasets/cifar100', train=False, download=True, transform=transform_test)
        wholedataset = torchvision.datasets.CIFAR100(root='/home/caifei/Project/Datasets/cifar100', train=True, download=True, transform=transform_test)
    elif args.dataset == 'tiny':
        cls_outdim = 200
        train_set_path = os.path.join('/home/caifei/Project/Datasets/tiny-imagenet-200', 'train')
        test_set_path = os.path.join('/home/caifei/Project/Datasets/tiny-imagenet-200', 'val')
        trainset = ImageFolder(root=train_set_path, transform=transform_train)
        testset = ImageFolder(root=test_set_path, transform=transform_test)
        wholedataset = ImageFolder(root=train_set_path, transform=transform_train)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)
    wholeloader = torch.utils.data.DataLoader(
        wholedataset, batch_size=1, shuffle=False, num_workers=1)
    #Model
    cls_indim = -10
    if args.model == 'resnet18':
        net = ResNet18(cls_indim, cls_outdim)
        if args.dataset == 'tiny':
            net = ResNetT('resnet18') #torchvision.models.resnet18(num_classes=200)
    elif args.model == 'resnet50':
        net = ResNet50(cls_indim, cls_outdim)
        if args.dataset == 'tiny':
            net = ResNetT('resnet50') #torchvision.models.resnet50(num_classes=200)
    elif args.model == 'senet50':
        net = SENet18(cls_indim, cls_outdim)
    elif args.model == 'mobilenetv2':
        net = MobileNetV2(cls_indim, cls_outdim)
    else:
        print('no valid network specified')
    return trainset,testloader,wholeloader,net
# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int, help="-1:No DistributedDataParallel,single machine and Gpu; 0:Yes DistributedDataParallel,Multi Gpus; >0: other auxiliary process")
    parser.add_argument("--dataset", default='cifar10', type=str, help="dataset:cifar10/cifar100/tiny/imagenet-1k/imagente-21k")
    parser.add_argument("--model", default='resnet18', type=str, help="model:resnet18/resnet50")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    world_size = torch.cuda.device_count()
    train(args.local_rank, world_size,ResNet18)