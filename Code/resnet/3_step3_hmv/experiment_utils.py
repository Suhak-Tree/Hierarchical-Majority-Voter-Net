import csv
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0

    def update(self, value, count=1):
        self.sum += float(value) * count
        self.count += count


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, classes=100, smoothing=0.1):
        super().__init__()
        self.classes = classes
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_prob = torch.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.full_like(log_prob, self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_prob, dim=1))


def accuracy(logits, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    result = []
    for k in topk:
        result.append(correct[:k].reshape(-1).float().sum().mul_(100.0 / target.size(0)))
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cifar100_loaders(data_root, batch_size, workers, download=False, train=True):
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )
    test_set = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=download,
        transform=test_transform,
    )
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **common)
    if not train:
        return None, test_loader

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    train_set = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=download,
        transform=train_transform,
    )
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **common)
    return train_loader, test_loader


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def strip_module_prefix(state_dict):
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def load_checkpoint(model, path, device="cpu", allow_missing_thresholds=False):
    checkpoint = torch.load(path, map_location=device)
    state_dict = strip_module_prefix(checkpoint_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if allow_missing_thresholds:
        missing = [key for key in missing if not key.endswith(("threshold_l1", "threshold_l2"))]
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint mismatch: missing={} unexpected={}".format(missing, unexpected)
        )
    return checkpoint


def save_checkpoint(state, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    latest = os.path.join(output_dir, "checkpoint.pth.tar")
    torch.save(state, latest)
    if is_best:
        torch.save(state, os.path.join(output_dir, "model_best.pth.tar"))


def write_rows_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_csv(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model
