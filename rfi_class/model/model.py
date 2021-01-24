import torch
from torch import nn
import torch.nn.functional as F
from rfi_class import config
import pytorch_lightning as pl
from torchvision import models


class RFIModelPretrained(pl.LightningModule):
    def __init__(self):
        super(RFIModelPretrained, self).__init__()
        n_classes = len(config.CLASSES)
        self.n_classes = n_classes
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1),
        )
        self.model = model
        self.accuracy = pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        out = self.model(x)
        return out

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, self.n_classes), target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        images = images.permute(0, 3, 1, 2)

        targets = batch["target"]
        targets = targets.type(torch.LongTensor).to(config.DEVICE)

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
        targets = targets.type(torch.LongTensor).to(config.DEVICE)

        out = self(images)
        loss = self.loss_fn(out, targets)
        accuracy = self.valid_accuracy(out, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss, accuracy

    def training_epoch_end(self, train_step_outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in train_step_outputs]).mean()
        print(f"Train Loss: {avg_val_loss:.2f}")

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.running_sanity_check:
            avg_val_loss = torch.tensor([x[0] for x in val_step_outputs]).mean()
            avg_val_acc = torch.tensor([x[1] for x in val_step_outputs]).mean()
            print(
                f"Epoch: {self.current_epoch} Val Acc: {avg_val_acc:.2f} Val Loss: {avg_val_loss:.2f} ",
                end="",
            )


class RFIModel(pl.LightningModule):
    def __init__(self):
        super(RFIModel, self).__init__()
        self.n_classes = len(config.CLASSES)
        pool_size = (2, 2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * pool_size[0] * pool_size[1], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.n_classes)
        self.maxpool = nn.AdaptiveMaxPool2d(pool_size)
        self.dropout = nn.Dropout(p=0.5)
        self.accuracy = pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        # C > M > R > B
        out = F.relu(self.maxpool(self.conv1(x)))
        out = self.bn1(out)
        out = F.relu(self.maxpool(self.conv2(out)))
        out = self.bn2(out)
        out = F.relu(self.maxpool(self.conv3(out)))
        out = self.bn3(out)
        out = F.relu(self.maxpool(self.conv4(out)))
        out = self.bn4(out)
        out = F.relu(self.maxpool(self.conv5(out)))
        out = self.bn5(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = F.relu(self.fc4(out))
        out = F.log_softmax(out)
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
        targets = targets.type(torch.LongTensor).to(config.DEVICE)

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
        targets = targets.type(torch.LongTensor).to(config.DEVICE)

        out = self(images)
        loss = self.loss_fn(out, targets)
        accuracy = self.valid_accuracy(out, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss, accuracy

    def training_epoch_end(self, train_step_outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in train_step_outputs]).mean()
        print(f"Train Loss: {avg_val_loss:.2f}")

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.running_sanity_check:
            avg_val_loss = torch.tensor([x[0] for x in val_step_outputs]).mean()
            avg_val_acc = torch.tensor([x[1] for x in val_step_outputs]).mean()
            print(
                f"Epoch: {self.current_epoch} Val Acc: {avg_val_acc:.2f} Val Loss: {avg_val_loss:.2f} ",
                end="",
            )