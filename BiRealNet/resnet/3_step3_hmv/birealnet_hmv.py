import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BiRealNet", "HardBinaryConv", "birealnet18"]


def binary_sign(x):
    """PopBin-compatible sign: zero is mapped to -1."""
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))


def round_ste(x):
    rounded = torch.round(x)
    return rounded.detach() - x.detach() + x


def threshold_sign_ste(score, normalizer):
    """Hard {-1, +1} forward with the Bi-Real polynomial surrogate backward."""
    hard = binary_sign(score)
    if not torch.is_grad_enabled():
        return hard

    x = score / normalizer.clamp_min(1.0)
    mask1 = x < -1
    mask2 = x < 0
    mask3 = x < 1
    out1 = -mask1.to(x.dtype) + (x * x + 2 * x) * (~mask1).to(x.dtype)
    out2 = out1 * mask2.to(x.dtype) + (-x * x + 2 * x) * (~mask2).to(x.dtype)
    smooth = out2 * mask3.to(x.dtype) + (~mask3).to(x.dtype)
    return hard.detach() - smooth.detach() + smooth


class BinaryActivation(nn.Module):
    def forward(self, x):
        hard = binary_sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = -mask1.to(x.dtype) + (x * x + 2 * x) * (~mask1).to(x.dtype)
        out2 = out1 * mask2.to(x.dtype) + (-x * x + 2 * x) * (~mask2).to(x.dtype)
        smooth = out2 * mask3.to(x.dtype) + (~mask3).to(x.dtype)
        return hard.detach() - smooth.detach() + smooth


class LearnableBias(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        return x + self.bias.expand_as(x)


class MajorityVoter(nn.Module):
    """Original PopBin hard forward and gradient approximation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x):
        hard = binary_sign(x)
        normalized = x / (self.in_channels * self.alpha.abs().clamp_min(1e-6))
        mask1 = normalized < -1
        mask2 = normalized < 0
        mask3 = normalized < 1
        out1 = -mask1.to(x.dtype) + (normalized * normalized + 2 * normalized) * (~mask1).to(x.dtype)
        out2 = out1 * mask2.to(x.dtype) + (-normalized * normalized + 2 * normalized) * (~mask2).to(x.dtype)
        smooth = out2 * mask3.to(x.dtype) + (~mask3).to(x.dtype)
        return hard.detach() - smooth.detach() + smooth


class HardBinaryConv(nn.Module):
    """PopBin binary convolution with optional two-level G-bit HMV."""

    layer_counter = 0

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        hmv_group_size=64,
        hmv_wiring="spatial",
        hmv_enabled=True,
        output_chunk=1,
    ):
        super().__init__()
        HardBinaryConv.layer_counter += 1
        self.layer_id = HardBinaryConv.layer_counter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001)
        self.majority_voter = MajorityVoter(in_channels, out_channels)

        if hmv_group_size <= 0:
            raise ValueError("hmv_group_size must be positive")
        if hmv_wiring not in ("spatial", "channel"):
            raise ValueError("hmv_wiring must be 'spatial' or 'channel'")
        self.hmv_group_size = int(hmv_group_size)
        self.hmv_wiring = hmv_wiring
        self.hmv_enabled = bool(hmv_enabled)
        self.output_chunk = max(1, int(output_chunk))

        reduction_length = in_channels * kernel_size * kernel_size
        self.group_count = math.ceil(reduction_length / self.hmv_group_size)
        self.threshold_l1 = nn.Parameter(torch.tensor(float(self.hmv_group_size // 2 + 1)))
        self.threshold_l2 = nn.Parameter(torch.tensor(float(self.group_count // 2 + 1)))

        self.collect_stats = False
        self.reset_stats()

    def reset_stats(self):
        self._stats = {
            "elements": 0,
            "matches": 0,
            "flat_positive": 0,
            "hmv_positive": 0,
            "l1_votes": 0,
            "l1_positive": 0,
            "l1_ties": 0,
        }

    def integer_thresholds(self):
        t1 = int(torch.round(self.threshold_l1.detach()).clamp(0, self.hmv_group_size + 1).item())
        t2 = int(torch.round(self.threshold_l2.detach()).clamp(0, self.group_count + 1).item())
        return t1, t2

    def clamp_thresholds_(self):
        with torch.no_grad():
            self.threshold_l1.clamp_(0, self.hmv_group_size + 1)
            self.threshold_l2.clamp_(0, self.group_count + 1)

    def _threshold_values(self):
        t1 = round_ste(self.threshold_l1.clamp(0, self.hmv_group_size + 1))
        t2 = round_ste(self.threshold_l2.clamp(0, self.group_count + 1))
        return t1, t2

    def _binary_weights(self):
        real_weights = self.weight
        scaling = real_weights.abs().mean(dim=(1, 2, 3), keepdim=True).permute(1, 0, 2, 3).detach()
        hard = torch.sign(real_weights)
        clipped = torch.clamp(real_weights, -1.0, 1.0)
        binary = hard.detach() - clipped.detach() + clipped
        return binary, scaling

    def _flat_forward(self, x, binary_weights):
        dot = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        return self.majority_voter(dot)

    def _unfold_input_and_validity(self, x):
        kh = kw = self.kernel_size
        unfolded = F.unfold(
            x,
            kernel_size=(kh, kw),
            padding=self.padding,
            stride=self.stride,
        )
        valid_source = x.new_ones((1, self.in_channels, x.size(2), x.size(3)))
        validity = F.unfold(
            valid_source,
            kernel_size=(kh, kw),
            padding=self.padding,
            stride=self.stride,
        )
        return unfolded, validity

    def _wire_vectors(self, unfolded, validity, weights):
        batch, reduction_length, positions = unfolded.shape
        out_channels = weights.size(0)
        kernel_area = self.kernel_size * self.kernel_size

        if self.hmv_wiring == "spatial":
            wired_input = unfolded
            wired_weights = weights.view(out_channels, reduction_length)
            wired_validity = validity
        else:
            wired_input = unfolded.view(batch, self.in_channels, kernel_area, positions)
            wired_input = wired_input.permute(0, 2, 1, 3).contiguous().view(batch, reduction_length, positions)
            wired_weights = weights.view(out_channels, self.in_channels, kernel_area)
            wired_weights = wired_weights.permute(0, 2, 1).contiguous().view(out_channels, reduction_length)
            wired_validity = validity.view(1, self.in_channels, kernel_area, positions)
            wired_validity = wired_validity.permute(0, 2, 1, 3).contiguous().view(1, reduction_length, positions)
        return wired_input, wired_validity, wired_weights

    def _hmv_forward(self, x, binary_weights):
        unfolded, validity = self._unfold_input_and_validity(x)
        reduction_length = unfolded.size(1)
        positions = unfolded.size(2)
        pad_bits = (-reduction_length) % self.hmv_group_size
        group_count = (reduction_length + pad_bits) // self.hmv_group_size
        if group_count != self.group_count:
            raise RuntimeError("unexpected HMV group count")

        out_h = (x.size(2) + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (x.size(3) + 2 * self.padding - self.kernel_size) // self.stride + 1
        threshold_l1, threshold_l2 = self._threshold_values()
        default_l1 = self.hmv_group_size // 2 + 1
        default_l2 = self.group_count // 2 + 1
        delta_l1 = threshold_l1 - default_l1
        delta_l2 = threshold_l2 - default_l2

        wired_input, wired_validity, wired_weights = self._wire_vectors(unfolded, validity, binary_weights)
        if pad_bits:
            wired_input = F.pad(wired_input, (0, 0, 0, pad_bits))
            wired_validity = F.pad(wired_validity, (0, 0, 0, pad_bits))
            wired_weights = F.pad(wired_weights, (0, pad_bits))

        grouped_input = wired_input.view(x.size(0), group_count, self.hmv_group_size, positions)
        grouped_weights = wired_weights.view(self.out_channels, group_count, self.hmv_group_size)
        signed_sum = torch.einsum("bgsl,ogs->bogl", grouped_input, grouped_weights)
        valid = wired_validity.view(1, group_count, self.hmv_group_size, positions).sum(dim=2).unsqueeze(1)
        match_count = (signed_sum + valid) * 0.5

        effective_l1 = torch.floor(valid * 0.5) + 1 + delta_l1
        effective_l1 = torch.minimum(torch.maximum(effective_l1, torch.zeros_like(effective_l1)), valid + 1)
        score_l1 = match_count - effective_l1 + 0.5
        alpha = self.majority_voter.alpha.abs().clamp_min(1e-6)
        normalizer_l1 = (valid * 0.5).clamp_min(1.0) * alpha
        vote_pm = threshold_sign_ste(score_l1, normalizer_l1)
        is_active = valid > 0
        positive_votes = torch.where(is_active, (vote_pm + 1) * 0.5, torch.zeros_like(vote_pm))

        active_group_count = is_active.sum(dim=2)
        effective_l2 = torch.floor(active_group_count * 0.5) + 1 + delta_l2
        effective_l2 = torch.minimum(
            torch.maximum(effective_l2, torch.zeros_like(effective_l2)),
            active_group_count + 1,
        )
        score_l2 = positive_votes.sum(dim=2) - effective_l2 + 0.5
        alpha_l2 = self.majority_voter.alpha[:, :, :, 0].abs().clamp_min(1e-6)
        normalizer_l2 = (active_group_count * 0.5).clamp_min(1.0) * alpha_l2
        hmv = threshold_sign_ste(score_l2, normalizer_l2)

        if self.collect_stats:
            with torch.no_grad():
                self._stats["l1_votes"] += int(is_active.sum().item()) * x.size(0) * self.out_channels
                self._stats["l1_positive"] += int(((vote_pm > 0) & is_active).sum().item())
                ties = (signed_sum == 0) & is_active
                self._stats["l1_ties"] += int(ties.sum().item())

        return hmv.view(x.size(0), self.out_channels, out_h, out_w)

    def forward(self, x):
        binary_weights, scaling = self._binary_weights()
        if not self.hmv_enabled:
            voted = self._flat_forward(x, binary_weights)
            return scaling * voted

        hmv = self._hmv_forward(x, binary_weights)
        if self.collect_stats:
            with torch.no_grad():
                flat = binary_sign(F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding))
                hmv_hard = binary_sign(hmv)
                self._stats["elements"] += flat.numel()
                self._stats["matches"] += int((flat == hmv_hard).sum().item())
                self._stats["flat_positive"] += int((flat > 0).sum().item())
                self._stats["hmv_positive"] += int((hmv_hard > 0).sum().item())
        return scaling * hmv

    def stats_row(self, name):
        t1, t2 = self.integer_thresholds()
        elements = self._stats["elements"]
        l1_votes = self._stats["l1_votes"]
        return OrderedDict(
            layer_id=self.layer_id,
            layer_name=name,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            reduction_length=self.in_channels * self.kernel_size * self.kernel_size,
            group_size=self.hmv_group_size,
            group_count=self.group_count,
            wiring=self.hmv_wiring,
            threshold_l1_raw=float(self.threshold_l1.detach().item()),
            threshold_l2_raw=float(self.threshold_l2.detach().item()),
            threshold_l1=t1,
            threshold_l2=t2,
            operator_match_rate=(self._stats["matches"] / elements if elements else float("nan")),
            mismatch_rate=(1 - self._stats["matches"] / elements if elements else float("nan")),
            flat_positive_rate=(self._stats["flat_positive"] / elements if elements else float("nan")),
            hmv_positive_rate=(self._stats["hmv_positive"] / elements if elements else float("nan")),
            l1_positive_rate=(self._stats["l1_positive"] / l1_votes if l1_votes else float("nan")),
            l1_tie_rate=(self._stats["l1_ties"] / l1_votes if l1_votes else float("nan")),
            samples=elements,
        )


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, hmv_config=None):
        super().__init__()
        config = dict(hmv_config or {})
        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride, **config)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.move1(out)
        out = self.prelu(out)
        return self.move2(out)


class BiRealNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, hmv_config=None):
        super().__init__()
        self.inplanes = 64
        self.hmv_config = dict(hmv_config or {})
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.Identity()
        HardBinaryConv.layer_counter = 0
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.hmv_config)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, hmv_config=self.hmv_config))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))

    def binary_convs(self):
        for name, module in self.named_modules():
            if isinstance(module, HardBinaryConv):
                yield name, module

    def set_hmv_enabled(self, enabled):
        for _, module in self.binary_convs():
            module.hmv_enabled = bool(enabled)

    def set_collect_stats(self, enabled, reset=True):
        for _, module in self.binary_convs():
            module.collect_stats = bool(enabled)
            if reset:
                module.reset_stats()

    def clamp_thresholds_(self):
        for _, module in self.binary_convs():
            module.clamp_thresholds_()

    def threshold_rows(self, include_stats=False):
        rows = []
        for name, module in self.binary_convs():
            row = module.stats_row(name)
            if not include_stats:
                keep = [
                    "layer_id", "layer_name", "in_channels", "out_channels", "kernel_size",
                    "reduction_length", "group_size", "group_count", "wiring",
                    "threshold_l1_raw", "threshold_l2_raw", "threshold_l1", "threshold_l2",
                ]
                row = OrderedDict((key, row[key]) for key in keep)
            rows.append(row)
        return rows


def birealnet18(**kwargs):
    return BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
