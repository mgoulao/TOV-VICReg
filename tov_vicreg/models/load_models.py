import os

import torch
from torch import nn
import torchvision.models as torchvision_models

import tov_vicreg.models.networks.vit as vits
from tov_vicreg.models.networks.resnet import ResnetCNN
import tov_vicreg.utils.pytorch_utils as pytorch_utils


def load_model(arch, pretrained_weights=None, patch_size=8, num_classes=0, freeze=False, n_channels=3, input_shape=(1, 3, 84, 84)):
    if arch.startswith("vit_"):
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=num_classes)
    elif arch == "sgi_resnet":
        model = ResnetCNN(input_channels=n_channels, depths=[64, 128, 256, 512], strides=[2,2,2,2])
    elif arch == "rainbow_cnn" or arch == "canonical":
        model = nn.Sequential(
            nn.Conv2d(n_channels, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
    elif arch == "data-efficient" or arch == "der":
        model = nn.Sequential(
            nn.Conv2d(n_channels, 32, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
    elif arch == "linear":
        model = nn.Sequential(nn.Linear(4, 128), nn.ReLU(),)
        model.num_features = 128
    else:
        model = torchvision_models.__dict__[arch]()

    if "vit" not in arch or "linear" not in arch:
        with torch.no_grad():
            model.num_features = model(torch.ones(input_shape)).shape[1]

    for p in model.parameters():
        p.requires_grad = not freeze

    if pretrained_weights is None or pretrained_weights == "_":
        print(f"No pretrained weights provided, using random {arch}")
        return model

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")

        if "teacher" in state_dict.keys():  # DINO
            state_dict = state_dict["teacher"]
        elif "state_dict" in state_dict.keys():  # MOCOv3
            state_dict = state_dict["state_dict"]
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith("base_encoder") and not k.startswith(
                    "base_encoder.head"
                ):
                    # remove prefix
                    state_dict[k[len("base_encoder.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif list(state_dict.keys())[0].startswith("vit."):
            for k in list(state_dict.keys()):
                if k.startswith("vit") and not k.startswith("vit.head"):
                    # remove prefix
                    state_dict[k[len("vit.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif list(state_dict.keys())[0].startswith("encoder."):
            for k in list(state_dict.keys()):
                if k.startswith("encoder") and not k.startswith("encoder.head"):
                    # remove prefix
                    state_dict[k[len("encoder.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove head
        for k in list(state_dict.keys()):
            if k.startswith("head."):
                del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                pretrained_weights, msg
            )
        )
    else:
        raise FileExistsError("Can't find file with pretrained weights")
    return model

