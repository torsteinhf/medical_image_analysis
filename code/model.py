import torch.nn as nn

from monai.networks.nets.densenet import DenseNet121
from monai.networks.nets import resnet34
from monai.networks.nets import resnet10

MODEL_TYPE = "densenet121" # densenet121, resnet34, resnet34pretrained, resnet10, resnet10pretrained

class ResNet34Pretrained(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = resnet34(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False,
            bias_downsample=True,
            shortcut_type="A"
        )
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.classifier(self.backbone(x))



class ResNet10Pretrained(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = resnet10(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False,
            bias_downsample=False,
            shortcut_type="B"
        )
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.classifier(self.backbone(x))



def get_model(in_channels, num_classes) -> nn.Module:
    if MODEL_TYPE == "densenet121":
        return DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
        )
    elif MODEL_TYPE == "resnet34":
        return resnet34(
            n_input_channels = in_channels,
            num_classes = num_classes,
        )
    elif MODEL_TYPE == "resnet34pretrained":
        return ResNet34Pretrained(num_classes=num_classes)
    elif MODEL_TYPE == "resnet10":
        return resnet10(
            n_input_channels = in_channels,
            num_classes = num_classes,
        )
    elif MODEL_TYPE == "resnet10pretrained":
        return ResNet10Pretrained(num_classes=num_classes)
    else:
        raise NotImplementedError("model not implemented")