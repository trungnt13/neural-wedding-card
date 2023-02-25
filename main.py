"""
Reference:
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

**Author**: `Alexis Jacq <https://alexis-jacq.github.io>`_
**Edited by**: `Winston Herring <https://github.com/winston6>`_
**Edited by**: `TrungNgo <https://github.com/trungnt13>`_
"""

import argparse
from collections import defaultdict
import copy
import os
from typing import List, Literal, Optional
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim
from torchmetrics.functional.image.tv import total_variation

description = """
Neural transfer using PyTorch.
"""

#########################################################
# Helper Functions
#########################################################
def download_image(url, filename: Optional[str] = None) -> str:
    if os.path.isfile(url):
        return url
    if filename is None:
        filename = os.path.join("/tmp", os.path.basename(url))
    if not os.path.exists(filename):
        urlretrieve(url, filename)
    return filename


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def is_gpu_available():
    return torch.cuda.is_available() or torch.backends.mps.is_available()


###################################################################
# Losses
###################################################################
class ContentLoss(nn.Module):
    """The content loss is a function that represents a weighted version of the content
    distance for an individual layer. The function takes the feature"""

    def __init__(self, target, loss: Literal["mse", "mae", "ssim"] = "mse"):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss_type = loss.lower().strip()

    def forward(self, input):
        if self.loss_type == "mae":
            self.loss = F.l1_loss(input, self.target)
        elif self.loss_type == "mse":
            self.loss = F.mse_loss(input, self.target)
        elif self.loss_type == "ssim":
            self.loss = (1 - ssim(input, self.target, data_range=(self.target.max() - self.target.min()).detach())) / 2
        else:
            raise ValueError(f"Invalid loss type {self.loss}")
        return input


def gram_matrix(input):
    """Style loss

    Note: the gram matrix must be normalized by dividing each element by
    the total number of elements in the matrix. This normalization is to
    counteract the fact that :math:`\hat{F}_{XL}` matrices with a large :math:`N` dimension yield
    larger values in the Gram matrix. These larger values will cause the
    first layers (before pooling layers) to have a larger impact during the
    gradient descent. Style features tend to be in the deeper layers of the
    network so this normalization step is crucial.

    Args:
        input (_type_): _description_

    Returns:
        _type_: _description_
    """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature, loss: Literal["mse", "mae", "ssim"] = "mse"):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss_type = loss.lower().strip()

    def forward(self, input):
        G = gram_matrix(input)
        if self.loss_type == "mae":
            loss = F.l1_loss(G, self.target)
        elif self.loss_type == "mse":
            loss = F.mse_loss(G, self.target)
        elif self.loss_type == "ssim":
            loss = (1 - ssim(G, self.target, data_range=(self.target.max() - self.target.min()).detach())) / 2
        else:
            raise ValueError(f"Invalid loss type {self.loss}")
        self.loss = loss
        return input


class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, input):
        self.loss = total_variation(input)
        return input


###################################################################
# The model, optimizer and losses
###################################################################
def get_model(name: str):
    if name == "vgg19":
        cnn = models.vgg19(pretrained=True).features
        content_layers_default = ["conv_4"]
        style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    elif name == "vgg16":
        cnn = models.vgg16(pretrained=True).features
        content_layers_default = ["conv_4"]
        style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    elif name == "vgg13":
        cnn = models.vgg13(pretrained=True).features
        content_layers_default = ["conv_4"]
        style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    else:
        raise ValueError(f"Invalid model name {name}")
    cnn = cnn.eval()
    tv_layers_default = content_layers_default + style_layers_default
    return cnn, content_layers_default, style_layers_default, tv_layers_default


def get_style_model_and_losses(
    cnn,
    style_img,
    content_img,
    content_layers: Optional[List[str]] = None,
    style_layers: Optional[List[str]] = None,
    tv_layers: Optional[List[str]] = None,
):
    if content_layers is None:
        content_layers = []
    if style_layers is None:
        style_layers = []
    if tv_layers is None:
        tv_layers = []

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []
    tv_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential()

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

        if name in tv_layers:
            # add total variation loss:
            tv_loss = TotalVariation()
            model.add_module("tv_loss_{}".format(i), tv_loss)
            tv_losses.append(tv_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[: (i + 1)]

    return model, style_losses, content_losses, tv_losses


def get_optimizer(input_img: torch.Tensor, lr: float = 1, opt: str = "lbfgs"):
    # this line to show that input is a parameter that requires a gradient
    if opt == "adam":
        optimizer = optim.Adam([input_img.requires_grad_()], lr=lr)
    elif opt == "lbfgs":
        optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr)
    elif opt == "sgd":
        optimizer = optim.SGD([input_img.requires_grad_()], lr=0.1)
    else:
        raise ValueError(f"Invalid optimizer {opt}")
    return optimizer


def get_transform(imsize: Optional[int] = None, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
    if imsize is None:
        imsize = 512 if is_gpu_available() else 128  # use small size if no gpu
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    loader = transforms.Compose(
        [
            transforms.Lambda(lambda ipath: Image.open(download_image(ipath)).convert("RGB")),
            transforms.Resize(imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
        ]
    )  # transform it into a torch tensor [B, C, H, W]

    unloader = transforms.Compose(
        [
            transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]),
            transforms.Lambda(lambda x: x.squeeze(0)),
            transforms.ToPILImage(),
        ]
    )
    return loader, unloader


###################################################################
# main
###################################################################
def run_style_transfer(
    cnn: str = "vgg19",
    style_img: str = "https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg",
    content_img: str = "https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg",
    optimizer: str = "adam",
    lr: float = 1e-3,
    num_steps: int = 300,
    style_weight: float = 1000000,
    content_weight: float = 1,
    tv_weight: float = 0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """Run the style transfer."""
    print("Building the style transfer model..")

    if device is None:
        device = get_device()

    loader, unloader = get_transform()
    style = loader(style_img).to(device)
    content = loader(content_img).to(device)

    cnn, content_layers, style_layers, tv_layers = get_model(cnn)
    cnn.to(device)

    model, style_losses, content_losses, tv_losses = get_style_model_and_losses(
        cnn,
        style_img=style,
        content_img=content,
        content_layers=content_layers,
        style_layers=style_layers,
        tv_layers=tv_layers,
    )
    model.eval()
    model.requires_grad_(False)

    content_var = content.clone().requires_grad_(True).to(device)
    optimizer = get_optimizer(content_var, opt=optimizer, lr=lr)

    if verbose:
        print(model)
        print(optimizer)

    losses = defaultdict(list)
    for step in range(num_steps):
        # correct the values of updated input image
        with torch.no_grad():
            content_var.clamp_(0, 1)

        optimizer.zero_grad()

        model(content_var)
        style_score = style_weight * sum(sl.loss for sl in style_losses)
        content_score = content_weight * sum(cl.loss for cl in content_losses)
        # tv_score = tv_weight * sum(tvl.loss for tvl in tv_losses)
        tv_score = torch.tensor(0.0, device=device)
        loss = style_score + content_score + tv_score
        loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(
                f"[{step}/{num_steps}] "
                f"Style-Loss:{style_score.item():4f} "
                f"Content-Loss:{content_score.item():4f} "
                f"TV-Loss:{tv_score.item():4f}"
            )
        # losses["style"].append(style_score.item())
        # losses["content"].append(content_score.item())
        # losses["tv"].append(tv_score.item())

    # a last correction...
    with torch.no_grad():
        content_var.clamp_(0, 1)
    content_var = content_var.cpu().squeeze(0).detach().numpy()

    return content_var


def main():
    run_style_transfer()


###################################################################
# Arguments
###################################################################

if __name__ == "__main__":
    main()
