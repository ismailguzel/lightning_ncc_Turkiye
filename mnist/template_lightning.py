import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as L

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._pytree")



class LitConvNet(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 1e-4)
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='maximum number of epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='the batch size')
    parser.add_argument('--accelerator', default='gpu', type=str,
                        help='accelerator to use')
    parser.add_argument('--strategy', default='ddp', type=str,
                        help='distributed strategy to use')
    args = parser.parse_args()


    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    dataset = MNIST('./data', download=False,
                    transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              num_workers=8, pin_memory=True)

    convnet = LitConvNet()

    trainer = L.Trainer(devices=args.gpus,
                        num_nodes=args.nodes,
                        max_epochs=args.epochs,
                        accelerator=args.accelerator,
                        strategy=args.strategy)

    from datetime import datetime
    t0 = datetime.now()
    trainer.fit(convnet, train_loader)
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))

    trainer.save_checkpoint("./out/lightning_model2.ckpt")


if __name__ == '__main__':
    main()
