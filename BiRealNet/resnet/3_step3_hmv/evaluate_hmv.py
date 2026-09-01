import argparse
import os
import time

import torch
import torch.nn as nn

from birealnet_hmv import birealnet18
from experiment_utils import (
    AverageMeter,
    accuracy,
    build_cifar100_loaders,
    load_checkpoint,
    set_seed,
    write_json,
    write_rows_csv,
)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate PopBin baseline and learned G64 HMV")
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference_checkpoint", required=True, help="2_step2 PopBin checkpoint")
    parser.add_argument("--hmv_checkpoint", required=True, help="trained stage-3 HMV checkpoint")
    parser.add_argument("--output", default="./hmv_eval")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--wiring", choices=("spatial", "channel"), default="spatial")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def make_model(args, enabled):
    return birealnet18(
        num_classes=100,
        hmv_config={
            "hmv_group_size": args.group_size,
            "hmv_wiring": args.wiring,
            "hmv_enabled": enabled,
        },
    )


@torch.no_grad()
def main():
    args = parse_args()
    print("args = {}".format(args), flush=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = {}".format(device), flush=True)

    reference = make_model(args, enabled=False)
    load_checkpoint(reference, args.reference_checkpoint, device="cpu", allow_missing_thresholds=True)
    reference.to(device).eval()

    adapted = make_model(args, enabled=True)
    checkpoint = load_checkpoint(adapted, args.hmv_checkpoint, device="cpu")
    adapted.to(device).eval()
    adapted.set_collect_stats(True, reset=True)

    _, loader = build_cifar100_loaders(
        args.data, args.batch_size, args.workers, args.download, train=False
    )
    criterion = nn.CrossEntropyLoss().to(device)
    metric_names = (
        "reference_loss", "reference_acc1", "reference_acc5",
        "adapted_flat_loss", "adapted_flat_acc1", "adapted_flat_acc5",
        "hmv_loss", "hmv_acc1", "hmv_acc5",
        "reference_adapted_flat_agreement", "reference_hmv_agreement",
        "adapted_flat_hmv_agreement",
    )
    meters = {name: AverageMeter() for name in metric_names}
    started = time.time()
    seen = 0

    for step, (images, target) in enumerate(loader):
        if args.max_batches and step >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch = images.size(0)

        reference_logits = reference(images)
        adapted.set_hmv_enabled(False)
        adapted_flat_logits = adapted(images)
        adapted.set_hmv_enabled(True)
        hmv_logits = adapted(images)

        for prefix, logits in (
            ("reference", reference_logits),
            ("adapted_flat", adapted_flat_logits),
            ("hmv", hmv_logits),
        ):
            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target)
            meters[prefix + "_loss"].update(loss.item(), batch)
            meters[prefix + "_acc1"].update(acc1.item(), batch)
            meters[prefix + "_acc5"].update(acc5.item(), batch)

        reference_pred = reference_logits.argmax(dim=1)
        adapted_flat_pred = adapted_flat_logits.argmax(dim=1)
        hmv_pred = hmv_logits.argmax(dim=1)
        meters["reference_adapted_flat_agreement"].update(
            (reference_pred == adapted_flat_pred).float().mean().item() * 100.0, batch
        )
        meters["reference_hmv_agreement"].update(
            (reference_pred == hmv_pred).float().mean().item() * 100.0, batch
        )
        meters["adapted_flat_hmv_agreement"].update(
            (adapted_flat_pred == hmv_pred).float().mean().item() * 100.0, batch
        )
        seen += batch
        if step % 20 == 0:
            print(
                "Test [{:03d}/{:03d}] reference {:.3f} adapted-flat {:.3f} HMV {:.3f}".format(
                    step,
                    len(loader),
                    meters["reference_acc1"].avg,
                    meters["adapted_flat_acc1"].avg,
                    meters["hmv_acc1"].avg,
                ),
                flush=True,
            )

    layer_rows = adapted.threshold_rows(include_stats=True)
    os.makedirs(args.output, exist_ok=True)
    write_rows_csv(os.path.join(args.output, "layer_operator_match.csv"), layer_rows)
    write_rows_csv(os.path.join(args.output, "learned_thresholds.csv"), adapted.threshold_rows())
    summary = {name: meter.avg for name, meter in meters.items()}
    summary.update(
        {
            "samples": seen,
            "elapsed_sec": time.time() - started,
            "group_size": args.group_size,
            "wiring": args.wiring,
            "reference_checkpoint": args.reference_checkpoint,
            "hmv_checkpoint": args.hmv_checkpoint,
            "hmv_checkpoint_epoch": checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown",
            "hmv_checkpoint_best_top1": checkpoint.get("best_top1_acc", "unknown") if isinstance(checkpoint, dict) else "unknown",
        }
    )
    write_json(
        os.path.join(args.output, "evaluation_summary.json"),
        {"summary": summary, "layers": layer_rows},
    )

    print("\nFinal evaluation", flush=True)
    print("  reference PopBin  acc1 {:.3f} acc5 {:.3f}".format(summary["reference_acc1"], summary["reference_acc5"]), flush=True)
    print("  adapted flat      acc1 {:.3f} acc5 {:.3f}".format(summary["adapted_flat_acc1"], summary["adapted_flat_acc5"]), flush=True)
    print("  learned HMV       acc1 {:.3f} acc5 {:.3f}".format(summary["hmv_acc1"], summary["hmv_acc5"]), flush=True)
    print("  reference/adapted-flat prediction agreement {:.3f}%".format(summary["reference_adapted_flat_agreement"]), flush=True)
    print("  reference/HMV prediction agreement {:.3f}%".format(summary["reference_hmv_agreement"]), flush=True)
    print("  adapted-flat/HMV prediction agreement {:.3f}%".format(summary["adapted_flat_hmv_agreement"]), flush=True)
    print("  wrote {}".format(os.path.abspath(args.output)), flush=True)


if __name__ == "__main__":
    main()
