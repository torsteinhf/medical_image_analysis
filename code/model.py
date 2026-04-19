import torch.nn as nn
from monai.networks.nets.densenet import DenseNet121


def get_model(in_channels: int = 1, num_classes: int = 1) -> nn.Module:
    """
    3D DenseNet121 from MONAI.
    num_classes=1 for binary (BCEWithLogitsLoss), or >1 for multi-class (CrossEntropyLoss).
    To swap architectures, replace DenseNet121 with e.g. EfficientNetBN, SEResNet50, etc.
    """
    return DenseNet121(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
    )
