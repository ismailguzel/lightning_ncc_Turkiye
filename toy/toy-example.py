import torch
import pytorch_lightning as L


class ToyExample(L.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model=model

    def training_step(self,batch):
        loss = self.model(batch).sum()
        return loss

    def configure_optimizers(self):
        torch.optim.Adam(self.model.parameters())

def main():
    # torch model
    model = torch.nn.Linear(32,2)
    # from pytorch to lightning
    pl_model = ToyExample(model)
    # your data
    train_loader = torch.utils.data.DataLoader(torch.randn(8,32))

    trainer = L.Trainer(devices=2, num_nodes=1, max_epochs=100, accelerator='gpu')

    trainer.fit(pl_model, train_loader)

if __name__ == "__main__":
    main()