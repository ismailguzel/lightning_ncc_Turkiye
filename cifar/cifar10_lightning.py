import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from gpu_utilization import MultiGPUUtilizationLogger
from pytorch_lightning.strategies import FSDPStrategy 


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._pytree")
os.makedirs("./out", exist_ok=True)

class LitResNet50(L.LightningModule):
    def __init__(self, learning_rate=1e-4, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained ResNet50 model
        weights_path = './model-weights/resnet50_weights.pth'
        model = resnet50(weights=None)

        # Load weights from local storage
        model.load_state_dict(torch.load(weights_path))
        
        self.model = model

        # Modify the final fully connected layer to match CIFAR-10 classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

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
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=2, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='maximum number of epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='the batch size')
    parser.add_argument('--accelerator', default='gpu', type=str,
                        help='accelerator to use')
    parser.add_argument('--strategy', default='ddp', type=str,
                        help='distributed strategy to use')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning rate')
    args = parser.parse_args()

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    transform = transforms.Compose([
    transforms.Resize(224),  # ResNet50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # Standard normalization for ResNet
    ])

    full_train_dataset = CIFAR10('./data', train=True, download=False, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = CIFAR10('./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)
    
    resnet_model = LitResNet50(learning_rate=args.learning_rate)

    logger = TensorBoardLogger(save_dir="./lightning_logs/", name="experiments", default_hp_metric=False)
    gpu_logger = MultiGPUUtilizationLogger(log_frequency=20)


    if args.strategy == "fsdp":
        strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",  
            cpu_offload=False
        )
    if args.strategy == "fsdp1":
        strategy = FSDPStrategy(
            sharding_strategy="SHARD_GRAD_OP",  
            cpu_offload=False
        )    
    if args.strategy == "fsdp2":
        strategy = FSDPStrategy(
            sharding_strategy="NO_SHARD", 
            cpu_offload=False
        )
    else:
        strategy = args.strategy 


    trainer = L.Trainer(
        devices=args.gpus,
        num_nodes=args.nodes,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        strategy=strategy,
        logger=logger,
        callbacks=[gpu_logger],
        log_every_n_steps=20 
    )

    from datetime import datetime
    t0 = datetime.now()
    trainer.fit(resnet_model, train_loader, val_loader)
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))

    model_name = "ResNet50"
    
    checkpoint_path = f"./out/lightning_{model_name}_{args.strategy}.ckpt"

    if os.path.exists(checkpoint_path):
        print(f"Existing checkpoint found at {checkpoint_path}, deleting...")
        os.remove(checkpoint_path)
        print("Old checkpoint deleted successfully.")

    trainer.save_checkpoint(checkpoint_path)
    trained_model = LitResNet50.load_from_checkpoint(checkpoint_path)

    test_trainer = L.Trainer(
        devices=1, 
        accelerator=args.accelerator,
        logger=logger 
    )

    print("Running test evaluation...")
    test_results = test_trainer.test(trained_model, test_loader)
    print(f"Test results: {test_results}")
    
    logger.log_hyperparams(
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "model_type": "ResNet",
            "gpus": args.gpus,
            "nodes": args.nodes,
            "strategy": strategy
        },
        metrics={
            "test_loss": test_results[0]["test_loss"],
            "test_acc": test_results[0]["test_acc"]
        }
    )

if __name__ == '__main__':
    main()
