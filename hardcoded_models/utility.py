from tqdm import tqdm

from hardcoded_models.ConvNeXt import register_convnext_models
from hardcoded_models.EfficientNet import register_efficientnet_models
from hardcoded_models.EfficientNetV2 import register_efficientnet_v2_models
from hardcoded_models.RegNetX import register_regnetx_models
from hardcoded_models.RegNetY import register_regnety_models
from hardcoded_models.ResNetRS import register_resnetrs_models


def register_hardcoded_models(pretrained: bool):
    registration_callbacks = [
        register_convnext_models,
        register_efficientnet_models,
        register_efficientnet_v2_models,
        register_regnetx_models,
        register_regnety_models,
        register_resnetrs_models
    ]
    for cb in registration_callbacks:
        cb(pretrained)