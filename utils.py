import os
import cv2
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
from einops import rearrange
from torchvision import transforms

from model import WatermarkModel


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]
    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate


def normalize(images):
    """
    Normalize an image array to [-1,1].
    """
    return ((images - 0.5) * 2).clamp(-1, 1)


def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)


def tensor_to_pil(images):
    images = images.permute(0, 2, 3, 1).float()
    if images.ndim == 3:
        images = images[None, ...]
    images = (denormalize(images) * 255).detach().cpu().numpy().round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def cv2_to_pil(img: np.ndarray):
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2_gray(img: Image):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return img


def pil_to_cv2(img: Image):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def pil_to_tensor(img: Image):
    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return image_transforms(img)


def save_image_for_tensor(image, save_path):
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    grid = 255. * rearrange(image, 'c h w -> h w c').cpu().detach().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(save_path)


def load_wm_model(ckpt_dir):
    wm_model_config_path = os.path.join(os.path.join(ckpt_dir, "wm_model_config.yaml"))
    wm_model_config = OmegaConf.load(wm_model_config_path)
    wm_model = WatermarkModel(**wm_model_config)
    wm_model_ckpt = torch.load(os.path.join(ckpt_dir, "wm_model.ckpt"), map_location="cpu")
    wm_model.load_state_dict(wm_model_ckpt)
    return wm_model
