from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


EXPECTED_GROUP_COUNT = 32
EXPECTED_CHANNEL_COUNT = 7552


@dataclass(frozen=True)
class ChannelIdentity:
    canonical_id: str
    group_id: str
    stage: str
    block: int
    hidden_conv: str
    local_channel_index: int


@dataclass
class StructuralGroup:
    group_id: str
    stage: str
    block: int
    hidden_conv: str
    producer_name: str
    norm_name: str
    consumer_name: str
    channels: int
    producer_module: Any
    norm_module: Any
    consumer_module: Any
    channels_identities: list[ChannelIdentity]


def _group_spec(hidden_conv: str) -> tuple[str, str, str]:
    if hidden_conv == "conv1":
        return "bn1", "conv2", "bottleneck_conv1_hidden"
    if hidden_conv == "conv2":
        return "bn2", "conv3", "bottleneck_conv2_hidden"
    raise ValueError(f"Unsupported hidden convolution: {hidden_conv}")


def discover_structural_groups(
    model: Any,
    include_stages: Sequence[str] = ("layer1", "layer2", "layer3", "layer4"),
    include_convs: Sequence[str] = ("conv1", "conv2"),
) -> list[StructuralGroup]:
    import torch.nn as nn

    modules = dict(model.named_modules())
    groups: list[StructuralGroup] = []
    prefix = "backbone.body."
    for stage in include_stages:
        stage_prefix = prefix + stage + "."
        block_indices = sorted(
            {
                int(parts[3])
                for name in modules
                if name.startswith(stage_prefix)
                for parts in [name.split(".")]
                if len(parts) >= 5 and parts[3].isdigit()
            }
        )
        for block in block_indices:
            for hidden_conv in include_convs:
                producer_name = f"{prefix}{stage}.{block}.{hidden_conv}"
                if producer_name not in modules:
                    continue
                norm_suffix, consumer_suffix, _ = _group_spec(hidden_conv)
                norm_name = f"{prefix}{stage}.{block}.{norm_suffix}"
                consumer_name = f"{prefix}{stage}.{block}.{consumer_suffix}"
                if norm_name not in modules or consumer_name not in modules:
                    raise RuntimeError(f"Missing dependency module for {producer_name}")
                producer = modules[producer_name]
                norm = modules[norm_name]
                consumer = modules[consumer_name]
                if not isinstance(producer, nn.Conv2d):
                    raise TypeError(f"Producer is not Conv2d: {producer_name}")
                if not isinstance(norm, nn.BatchNorm2d):
                    raise TypeError(f"Normalization is not BatchNorm2d: {norm_name}")
                if not isinstance(consumer, nn.Conv2d):
                    raise TypeError(f"Consumer is not Conv2d: {consumer_name}")
                if producer.out_channels != norm.num_features:
                    raise RuntimeError(f"Producer/BN mismatch at {producer_name}")
                if producer.out_channels != consumer.in_channels:
                    raise RuntimeError(f"BN/consumer mismatch at {producer_name}")
                group_id = f"{stage}.{block}.{hidden_conv}"
                identities = [
                    ChannelIdentity(
                        canonical_id=f"{producer_name}|channel={index}",
                        group_id=group_id,
                        stage=stage,
                        block=block,
                        hidden_conv=hidden_conv,
                        local_channel_index=index,
                    )
                    for index in range(int(producer.out_channels))
                ]
                groups.append(
                    StructuralGroup(
                        group_id=group_id,
                        stage=stage,
                        block=block,
                        hidden_conv=hidden_conv,
                        producer_name=producer_name,
                        norm_name=norm_name,
                        consumer_name=consumer_name,
                        channels=int(producer.out_channels),
                        producer_module=producer,
                        norm_module=norm,
                        consumer_module=consumer,
                        channels_identities=identities,
                    )
                )

    if tuple(include_stages) == ("layer1", "layer2", "layer3", "layer4") and tuple(include_convs) == ("conv1", "conv2"):
        if len(groups) != EXPECTED_GROUP_COUNT:
            raise RuntimeError(f"Expected {EXPECTED_GROUP_COUNT} structural groups, found {len(groups)}")
        channels = [identity.canonical_id for group in groups for identity in group.channels_identities]
        if len(channels) != EXPECTED_CHANNEL_COUNT or len(set(channels)) != EXPECTED_CHANNEL_COUNT:
            raise RuntimeError(f"Expected {EXPECTED_CHANNEL_COUNT} unique channels, found {len(set(channels))}")
    return groups
