import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from gpu_utilization import MultiGPUUtilizationLogger


class LitConvNet(L.LightningModule):
    def __init__(self, learning_rate=1e-4, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

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

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        opt = self.optimizers()
        self.log("learning_rate", opt.param_groups[0]["lr"], on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.log("test_acc", acc, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),  lr=self.hparams.learning_rate)
        return optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=2, type=int, metavar='N',
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
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning rate')
    args = parser.parse_args()

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    full_train_dataset = MNIST('./data', train=True, download=False, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = MNIST('./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)

    convnet = LitConvNet(learning_rate=args.learning_rate)

    logger = TensorBoardLogger(save_dir="./lightning_logs/", name="mnist", default_hp_metric=False)
    gpu_logger = MultiGPUUtilizationLogger()

    trainer = L.Trainer(
        devices=args.gpus,
        num_nodes=args.nodes,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        strategy=args.strategy,
        logger=logger,
        callbacks=[gpu_logger],
        log_every_n_steps=20
    )

    from datetime import datetime
    t0 = datetime.now()
    trainer.fit(convnet, train_loader, val_loader)
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))

    print("Running test evaluation...")
    test_results = trainer.test(convnet, test_loader)
    print(f"Test results: {test_results}")
    
    logger.log_hyperparams(
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "model_type": "ConvNet",
            "gpus": args.gpus,
            "nodes": args.nodes,
            "strategy": args.strategy
        },
        metrics={
            "test_loss": test_results[0]["test_loss"],
            "test_acc": test_results[0]["test_acc"]
        }
    )

if __name__ == '__main__':
    main()
