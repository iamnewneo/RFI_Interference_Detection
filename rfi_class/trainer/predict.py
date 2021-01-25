import torch
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl

from rfi_class import config


def loss_fn(out, target):
    n_classes = len(config.CLASSES)
    return nn.CrossEntropyLoss()(out.view(-1, n_classes), target)


def evaluate_model(model, data_loader):
    all_ids = []

    all_targets = torch.FloatTensor([])
    all_predictions = torch.FloatTensor([])

    model = model.to(config.DEVICE)
    result = {}
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"]
            images = images.permute(0, 3, 1, 2)
            images = images.to(config.DEVICE)

            ids = batch["id"]

            targets = batch["target"]
            targets = targets.type(torch.LongTensor)
            out = model(images)
            out = out.to("cpu")
            loss = loss_fn(out, targets)
            _, predicted = torch.max(out.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            total_loss += loss.item()
            # print(
            #     f"Batch Loss: {loss} Batch Accuracy: {test_accuracy(predicted, targets)}"
            # )
            all_targets = torch.cat((all_targets, targets), axis=0)
            all_predictions = torch.cat((all_predictions, predicted), axis=0)
            all_ids = all_ids + ids

    result["ids"] = all_ids
    result["targets"] = all_targets
    result["predictions"] = all_predictions
    result["accuracy"] = (correct * 100) / total
    result["loss"] = total_loss / len(data_loader)
    return result