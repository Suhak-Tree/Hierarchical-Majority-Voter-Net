import argparse
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from birealnet_hmv import birealnet18
from experiment_utils import (
    AverageMeter,
    LabelSmoothingCrossEntropy,
    accuracy,
    append_csv,
    build_cifar100_loaders,
    checkpoint_state_dict,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    strip_module_prefix,
    unwrap,
    write_json,
    write_rows_csv,
)


def parse_args():
    parser = argparse.ArgumentParser("Bi-RealNet PopBin -> G64 HMV stage-3 QAT")
    parser.add_argument("--data", required=True, help="CIFAR-100 root containing cifar-100-python")
    parser.add_argument("--resume", required=True, help="trained 2_step2 PopBin checkpoint")
    parser.add_argument("--hmv_resume", default="", help="optional stage-3 checkpoint to continue")
    parser.add_argument(
        "--continue_run",
        action="store_true",
        help="also restore epoch/optimizer/scheduler; omit when moving threshold stage to full stage",
    )
    parser.add_argument("--save", default="./models_hmv_g64_spatial")
    parser.add_argument("--objective", choices=("kd", "ce"), default="kd")
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument("--stage", choices=("threshold", "full"), default="threshold")
    parser.add_argument("--train_l1_threshold", type=int, choices=(0, 1), default=1)
    parser.add_argument("--train_l2_threshold", type=int, choices=(0, 1), default=1)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--wiring", choices=("spatial", "channel"), default="spatial")
    parser.add_argument("--output_chunk", type=int, default=1, help="reserved for interface compatibility")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    return parser.parse_args()


def make_model(args):
    return birealnet18(
        num_classes=100,
        hmv_config={
            "hmv_group_size": args.group_size,
            "hmv_wiring": args.wiring,
            "hmv_enabled": True,
            "output_chunk": args.output_chunk,
        },
    )


def configure_trainable_parameters(model, args):
    if args.stage == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False

    for _, module in model.binary_convs():
        module.threshold_l1.requires_grad = bool(args.train_l1_threshold)
        module.threshold_l2.requires_grad = bool(args.train_l2_threshold)

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters selected")
    return trainable


def build_optimizer(model, args):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    decay = [p for name, p in trainable if p.ndim == 4 or "conv" in name]
    decay_ids = {id(p) for p in decay}
    no_decay = [p for _, p in trainable if id(p) not in decay_ids]
    groups = []
    if no_decay:
        groups.append({"params": no_decay})
    if decay:
        groups.append({"params": decay, "weight_decay": args.weight_decay})
    return torch.optim.Adam(groups, lr=args.learning_rate)


def build_teacher(path, device):
    if not path:
        raise ValueError("--teacher_checkpoint is required when --objective kd")
    from pytorch_cifar100.models import resnet

    teacher = resnet.resnet18()
    checkpoint = torch.load(path, map_location=device)
    state = strip_module_prefix(checkpoint_state_dict(checkpoint))
    incompatible = teacher.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "teacher checkpoint mismatch: missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def distribution_loss(student_logits, teacher_logits):
    teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)
    return torch.mean(torch.sum(-teacher_prob * torch.log_softmax(student_logits, dim=1), dim=1))


def run_train_epoch(loader, model, teacher, criterion_ce, optimizer, device, epoch, args):
    model.train()
    if args.stage == "threshold":
        # Threshold-only means exactly that: do not silently adapt BatchNorm buffers.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    if teacher is not None:
        teacher.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    started = time.time()

    for step, (images, target) in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        if args.objective == "kd":
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss = distribution_loss(logits, teacher_logits)
        else:
            loss = criterion_ce(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        unwrap(model).clamp_thresholds_()

        acc1, acc5 = accuracy(logits, target)
        batch = images.size(0)
        loss_meter.update(loss.item(), batch)
        top1_meter.update(acc1.item(), batch)
        top5_meter.update(acc5.item(), batch)
        if step % args.print_freq == 0:
            print(
                "Epoch [{:03d}] [{:04d}/{:04d}] loss {:.4f} acc1 {:.3f} acc5 {:.3f}".format(
                    epoch, step, len(loader), loss_meter.avg, top1_meter.avg, top5_meter.avg
                ),
                flush=True,
            )
    return {
        "loss": loss_meter.avg,
        "acc1": top1_meter.avg,
        "acc5": top5_meter.avg,
        "elapsed_sec": time.time() - started,
    }


@torch.no_grad()
def validate(loader, model, criterion, device, args):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    for step, (images, target) in enumerate(loader):
        if args.max_val_batches and step >= args.max_val_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, target)
        acc1, acc5 = accuracy(logits, target)
        batch = images.size(0)
        loss_meter.update(loss.item(), batch)
        top1_meter.update(acc1.item(), batch)
        top5_meter.update(acc5.item(), batch)
    print("Validation acc1 {:.3f} acc5 {:.3f}".format(top1_meter.avg, top5_meter.avg), flush=True)
    return {"loss": loss_meter.avg, "acc1": top1_meter.avg, "acc5": top5_meter.avg}


def export_thresholds(model, output_dir, epoch):
    rows = unwrap(model).threshold_rows()
    write_rows_csv(os.path.join(output_dir, "thresholds_latest.csv"), rows)
    write_json(
        os.path.join(output_dir, "thresholds_latest.json"),
        {"epoch": epoch, "layers": rows},
    )


def main():
    args = parse_args()
    print("args = {}".format(args), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for full HMV QAT")
    device = torch.device("cuda")
    set_seed(args.seed)
    cudnn.benchmark = True

    model = make_model(args)
    stage2 = load_checkpoint(model, args.resume, device="cpu", allow_missing_thresholds=True)
    print(
        "loaded 2_step2 checkpoint epoch={} best_top1_acc={}".format(
            stage2.get("epoch", "unknown") if isinstance(stage2, dict) else "unknown",
            stage2.get("best_top1_acc", "unknown") if isinstance(stage2, dict) else "unknown",
        ),
        flush=True,
    )
    start_epoch = 0
    best_acc1 = 0.0
    configure_trainable_parameters(model, args)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda completed: max(0.0, 1.0 - completed / max(1, args.epochs))
    )

    if args.hmv_resume:
        checkpoint = load_checkpoint(model, args.hmv_resume, device="cpu")
        if args.continue_run:
            start_epoch = int(checkpoint.get("epoch", 0))
            best_acc1 = float(checkpoint.get("best_top1_acc", 0.0))
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            print("continued HMV run at epoch {}".format(start_epoch), flush=True)
        else:
            print("initialized new stage from HMV checkpoint; optimizer and epoch reset", flush=True)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    teacher = build_teacher(args.teacher_checkpoint, device) if args.objective == "kd" else None
    train_loader, val_loader = build_cifar100_loaders(
        args.data, args.batch_size, args.workers, args.download, train=True
    )
    criterion_ce = LabelSmoothingCrossEntropy(100, args.label_smooth).to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)
    os.makedirs(args.save, exist_ok=True)
    export_thresholds(model, args.save, start_epoch)

    for epoch in range(start_epoch, args.epochs):
        print("learning_rate: {}".format(optimizer.param_groups[0]["lr"]), flush=True)
        train_result = run_train_epoch(
            train_loader, model, teacher, criterion_ce, optimizer, device, epoch, args
        )
        val_result = validate(val_loader, model, criterion_val, device, args)
        scheduler.step()
        is_best = val_result["acc1"] > best_acc1
        best_acc1 = max(best_acc1, val_result["acc1"])
        state = {
            "epoch": epoch + 1,
            "state_dict": unwrap(model).state_dict(),
            "best_top1_acc": best_acc1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hmv_config": {
                "group_size": args.group_size,
                "wiring": args.wiring,
                "stage": args.stage,
                "objective": args.objective,
            },
            "args": vars(args),
        }
        save_checkpoint(state, args.save, is_best=is_best)
        export_thresholds(model, args.save, epoch + 1)
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "train_acc1": train_result["acc1"],
            "train_acc5": train_result["acc5"],
            "val_loss": val_result["loss"],
            "val_acc1": val_result["acc1"],
            "val_acc5": val_result["acc5"],
            "best_acc1": best_acc1,
            "elapsed_sec": train_result["elapsed_sec"],
        }
        append_csv(os.path.join(args.save, "training_history.csv"), row)
        print(
            "epoch {} val_acc1 {:.3f} best {:.3f}".format(epoch + 1, val_result["acc1"], best_acc1),
            flush=True,
        )


if __name__ == "__main__":
    main()
