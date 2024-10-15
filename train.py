import os
import argparse
import datetime
import json
import logging

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from kornia.metrics import psnr, ssim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from custom_insp2p import CustomStableDiffusionInstructPix2PixPipeline
from dataset import get_hugging_instruct_pix2pix_dataset, collate_fn
from model import WatermarkModel
from utils import (
    decoded_message_error_rate_batch,
    denormalize,
)

logger = logging.getLogger(__name__)


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args_dict = vars(args)
    output_with_time_dir = os.path.join(args.output_dir, now)
    os.makedirs(output_with_time_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%m/%d/%Y %H:%M:%S")
    fhlr = logging.FileHandler(os.path.join(output_with_time_dir, "log.txt"))
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    wm_model_config = OmegaConf.load(args.wm_model_config)
    args.message_length = wm_model_config["wm_enc_config"]["message_length"]
    wm_model = WatermarkModel(
        **wm_model_config,
        device=device,
        weight_dtype=weight_dtype,
    )

    params_to_optimize = list(p for p in wm_model.parameters() if p.requires_grad)

    pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=weight_dtype,
        local_files_only=True
    ).to(device)
    wm_model.train()
    pipe.freeze_params()
    pipe.text_encoder.train()
    pipe.unet.train()
    pipe.vae.train()

    train_dataset = get_hugging_instruct_pix2pix_dataset(args.train_data_dir, args.image_size, accelerator)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    opt = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate,)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    wm_model, opt, train_dataloader, lr_scheduler = accelerator.prepare(
        wm_model, opt, train_dataloader, lr_scheduler
    )

    def save_all(g_model, save_dir):
        unwrapped_model = accelerator.unwrap_model(g_model)
        accelerator.save(unwrapped_model.state_dict(), os.path.join(save_dir, "wm_model.ckpt"))
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(args_dict, f, indent=2)
        OmegaConf.save(wm_model_config, os.path.join(save_dir, "wm_model_config.yaml"))

    step = 0
    global_step = 0
    finished_flag = False
    while True:
        for data in train_dataloader:
            step += 1
            with accelerator.accumulate(wm_model):
                message = torch.randint(0, 2, (args.batch_size, args.message_length)).to(
                    device=device, dtype=torch.float32
                )
                image, prompt = data["image"], data["prompt"]
                wm_image = wm_model.encoder(image, message)

                image_latents = pipe.vae.encode(image.to(dtype=weight_dtype)).latent_dist.mode()
                wm_image_latents = pipe.vae.encode(wm_image.to(dtype=weight_dtype)).latent_dist.mode()

                decoded_message_before_edit = wm_model.decoder(wm_image.to(dtype=torch.float32))

                generated_image = pipe(
                    prompt, image=wm_image, num_images_per_prompt=1, num_inference_steps=20,
                    guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=args.last_grad_steps,
                    output_type="pt",
                )
                decoded_message_after_edit = wm_model.decoder(generated_image.to(dtype=torch.float32))

                # Calculate the total loss
                enc_pixel_loss = F.mse_loss(image.float(), wm_image.float())
                enc_latent_loss = F.mse_loss(image_latents.float(), wm_image_latents.float())
                dec_loss_before_edit = F.mse_loss(message, decoded_message_before_edit)
                dec_loss_after_edit = F.mse_loss(message, decoded_message_after_edit)
                enc_loss = enc_pixel_loss + args.enc_latent_weight * enc_latent_loss
                dec_loss = dec_loss_before_edit + args.decoder_weight * dec_loss_after_edit
                loss = enc_loss + dec_loss

                accelerator.backward(loss)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.log_steps == 0:
                        psnr_value = psnr(denormalize(wm_image.detach()), denormalize(image), 1)
                        ssim_value = torch.mean(ssim(denormalize(wm_image.detach()), denormalize(image), window_size=5))
                        error_rate_after_edit = decoded_message_error_rate_batch(
                            message, decoded_message_after_edit
                        )
                        error_rate_before_edit = decoded_message_error_rate_batch(
                            message, decoded_message_before_edit
                        )
                        log_dict = {
                            "step": step,
                            "global_step": global_step,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "enc_pixel_loss": enc_pixel_loss.detach().item(),
                            "enc_latent_loss": enc_latent_loss.detach().item(),
                            "dec_loss_before_edit": dec_loss_before_edit.detach().item(),
                            "dec_loss_after_edit": dec_loss_after_edit.detach().item(),
                            "psnr": psnr_value.item(),
                            "ssim": ssim_value.item(),
                            "error_rate_before_edit": error_rate_before_edit,
                            "error_rate": error_rate_after_edit,
                        }
                        logger.info(log_dict)

                    if global_step % args.save_steps == 0:
                        save_step_dir = os.path.join(output_with_time_dir, f"step{global_step}")
                        os.makedirs(save_step_dir, exist_ok=True)
                        logger.info("save models!")
                        save_all(wm_model, save_step_dir)

            if global_step >= args.max_train_steps:
                finished_flag = True
                break

        if finished_flag:
            break

    if accelerator.is_main_process:
        save_all(wm_model, output_with_time_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--message_length", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--decoder_weight", type=float, default=1.5)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--wm_model_config", type=str, default=None)
    parser.add_argument("--last_grad_steps", type=int, default=3)
    parser.add_argument("--enc_latent_weight", type=float, default=None)
    args = parser.parse_args()
    main(args)
