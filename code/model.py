import torch.nn as nn
from monai.networks.nets.densenet import DenseNet121

# 3D DenseNet121 from MONAI.
# num_classes = 1 for binary (BCEWithLogitsLoss)
# num_classes > 1 for multi-class (CrossEntropyLoss).

def get_model(in_channels: int = 1, num_classes: int = 1) -> nn.Module:
    return DenseNet121(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
    )