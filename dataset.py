from functools import partial

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def preprocess_images(examples, image_size):
    train_transforms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
    )
    original_images = np.concatenate(
        [convert_to_np(image, image_size) for image in examples["original_image"]]
    )
    images = torch.tensor(original_images)
    images = 2 * (images / 255) - 1
    return train_transforms(images)


def preprocess_train(examples, image_size):
    original_images = preprocess_images(examples, image_size)
    original_images = original_images.reshape(-1, 3, image_size, image_size)
    examples["image"] = original_images
    examples["prompt"] = list(examples["edit_prompt"])
    return examples


def collate_fn(examples):
    image = torch.stack([example["image"] for example in examples])
    image = image.to(memory_format=torch.contiguous_format).float()
    prompt = [example["prompt"] for example in examples]
    return {"image": image, "prompt": prompt}


def get_hugging_instruct_pix2pix_dataset(instance_data_root, image_size, accelerator):
    dataset = load_dataset(instance_data_root)
    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(partial(preprocess_train, image_size=image_size))
    return train_dataset
