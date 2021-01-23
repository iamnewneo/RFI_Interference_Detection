import torch
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl

from rfi_class import config


def loss_fn(out, target):
    n_classes = len(config.CLASSES)
    return nn.CrossEntropyLoss()(out.view(-1, n_classes), target)


def evaluate_model(model, data_loader):
    test_accuracy = pl.metrics.Accuracy()
    all_ids = []
    all_losses = torch.FloatTensor([])
    all_targets = torch.FloatTensor([])
    all_predictions = torch.FloatTensor([])
    result = {}
    for batch in data_loader:
        images = batch["image"]
        images = images.permute(0, 3, 1, 2)

        ids = batch["id"]

        targets = batch["target"]
        targets = targets.type(torch.LongTensor).to(config.DEVICE)

        out = model(images)
        loss = loss_fn(out, targets)
        _, predicted = torch.max(out.data, 1)

        all_targets = torch.cat((all_targets, targets), axis=0)
        all_predictions = torch.cat((all_predictions, predicted), axis=0)
        all_losses = torch.cat((all_losses, torch.FloatTensor([loss])), axis=0)
        all_ids = all_ids + ids

    overall_acc = test_accuracy(all_predictions, all_targets)
    result["ids"] = all_ids
    result["predictions"] = all_predictions
    result["accuracy"] = overall_acc
    result["loss"] = all_losses.mean()
    return result
