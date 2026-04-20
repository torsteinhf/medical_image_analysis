import torch.nn as nn

from monai.networks.nets.densenet import DenseNet121


MODEL_TYPE = "dense" # resnet34

def get_model(in_channels, num_classes) -> nn.Module:
    return DenseNet121(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
    )


from monai.networks.nets import resnet34

# def get_model(in_channels, num_classes) -> nn.Module:
#     return resnet34(
#         n_input_channels = in_channels,
#         num_classes = num_classes,
#     )

# class ResNet34Pretrained(nn.Module):
#     def __init__(self, num_classes: int = 3):
#         super().__init__()
#         self.backbone = resnet34(
#             pretrained=True,
#             spatial_dims=3,
#             n_input_channels=1,
#             feed_forward=False,
#             bias_downsample=True,
#             shortcut_type="B"
#         )
#         self.classifier = nn.Linear(512, num_classes)
        
#     def forward(self, x):
#         return self.classifier(self.backbone(x))

# def get_model(in_channels: int = 1, num_classes: int = 3) -> nn.Module:
#     return ResNet34Pretrained(num_classes=num_classes)

# from monai.networks.nets import resnet10

# def get_model(in_channels, num_classes) -> nn.Module:
#     return resnet10(
#         n_input_channels = in_channels,
#         num_classes = num_classes,
#     )


