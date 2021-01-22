import torch
from torch import nn
import torch.nn.functional as F
from rfi_class import config
import pytorch_lightning as pl


class RFIModel(pl.LightningModule):
    def __init__(self):
        super(RFIModel, self).__init__()
        self.n_classes = len(config.CLASSES)
        pool_size = (2, 2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool = nn.AdaptiveMaxPool2d(pool_size)
        self.fc1 = nn.Linear(64 * pool_size[0] * pool_size[1], 500)
        self.fc2 = nn.Linear(500, self.n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, self.n_classes), target)

    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        images = images.permute(0, 3, 1, 2)

        targets = batch["target"]
        targets = targets.type(torch.LongTensor)

        out = self(images)
        loss = self.loss_fn(out, targets)
        accuracy = self.train_accuracy(out, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        images = images.permute(0, 3, 1, 2)

        targets = batch["target"]
        targets = targets.type(torch.LongTensor)

        out = self(images)
        loss = self.loss_fn(out, targets)
        accuracy = self.valid_accuracy(out, targets)
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", accuracy, prog_bar=True)
        return loss, accuracy

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x[0] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x[1] for x in val_step_outputs]).mean()
        self.log("avg_val_loss", avg_val_loss, on_epoch=True, prog_bar=True)
        self.log("avg_val_acc", avg_val_acc, on_epoch=True, prog_bar=True)


# class RFINet(nn.Module):
#     def __init__(self, pool_size=(4, 4)):
#         super(RFINet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.maxpool = nn.AdaptiveMaxPool2d(pool_size)
#         self.fc1 = nn.Linear(64 * pool_size[0] * pool_size[1], 500)
#         self.fc2 = nn.Linear(500, 4)

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         out = F.relu(self.conv3(out))
#         out = self.maxpool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out