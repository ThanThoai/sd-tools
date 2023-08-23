#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import omegaconf

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


logger = get_logger(__name__)


def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- textual_inversion
inference: true
---
    """
    model_card = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def log_validation(
    text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch
):
    logger.info(
        f"Running validation... \n Generating {args.validate.num_validation_images} images with prompt:"
        f" {args.validate.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.training.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        None
        if args.training.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.training.seed)
    )
    images = []
    for _ in range(args.validate.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(
                args.validate.validation_prompt,
                num_inference_steps=25,
                generator=generator,
            ).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, epoch, dataformats="NHWC"
            )

    del pipeline
    torch.cuda.empty_cache()
    return images


def save_progress(
    text_encoder,
    placeholder_token_ids,
    accelerator,
    args,
    save_path,
    safe_serialization=True,
):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {
        args.dataset.placeholder_token: learned_embeds.detach().cpu()
    }

    if safe_serialization:
        safetensors.torch.save_file(
            learned_embeds_dict, save_path, metadata={"format": "pt"}
        )
    else:
        torch.save(learned_embeds_dict, save_path)


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        no_template=False,
        use_caption=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.no_template = no_template
        self.use_caption = use_caption

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        img_path = self.image_paths[i % self.num_images]
        image = Image.open(img_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.no_template:
            text = placeholder_string
        elif self.use_caption:
            caption_path = os.path.join(
                "datas/caption", img_path.split("/")[-1].replace(".jpg", ".caption")
            )
            text = open(caption_path, "r").read().strip()
        else:
            text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
            ]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def main(config_file):
    args = omegaconf.OmegaConf.load(config_file)
    logging_dir = os.path.join(args.training.output_dir, args.training.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.training.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        mixed_precision=args.training.mixed_precision,
        log_with=args.training.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.training.seed is not None:
        set_seed(args.training.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.training.output_dir is not None:
            os.makedirs(args.training.output_dir, exist_ok=True)

    # Load tokenizer
    if args.training.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.training.tokenizer_name)
    elif args.training.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.training.pretrained_model_name_or_path, subfolder="tokenizer"
        )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.training.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.training.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.training.pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.training.pretrained_model_name_or_path,
        subfolder="unet",
    )

    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.dataset.placeholder_token]

    if args.dataset.num_vectors < 1:
        raise ValueError(
            f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}"
        )

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.dataset.num_vectors):
        additional_tokens.append(f"{args.dataset.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.dataset.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.dataset.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(
        args.dataset.initializer_token, add_special_tokens=False
    )
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.training.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.training.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.training.scale_lr:
        args.training.learning_rate = (
            args.training.learning_rate
            * args.training.gradient_accumulation_steps
            * args.training.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.training.learning_rate,
        betas=(args.training.adam_beta1, args.training.adam_beta2),
        weight_decay=args.training.adam_weight_decay,
        eps=args.training.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.dataset.train_data_dir,
        tokenizer=tokenizer,
        size=args.training.resolution,
        placeholder_token=(
            " ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))
        ),
        repeats=args.dataset.repeats,
        learnable_property=args.dataset.learnable_property,
        center_crop=args.dataset.center_crop,
        set="train",
        no_template=args.dataset.no_template,
        use_caption=args.dataset.use_caption,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.training.train_batch_size,
        shuffle=True,
        num_workers=args.training.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training.gradient_accumulation_steps
    )
    # if args.training.max_train_steps is None:
    #     args.training.max_train_steps = (
    #         args.num_train_epochs * num_update_steps_per_epoch
    #     )
    #     overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.training.max_train_steps * accelerator.num_processes,
        num_cycles=args.training.lr_num_cycles,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training.gradient_accumulation_steps
    )
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        args.training.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion")

    # Train!
    total_batch_size = (
        args.training.train_batch_size
        * accelerator.num_processes
        * args.training.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.training.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.training.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {args.training.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.training.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )

    for epoch in range(first_epoch, num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            # if (
            #     args.resume_from_checkpoint
            #     and epoch == first_epoch
            #     and step < resume_step
            # ):
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue

            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = (
                    vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
                    .latent_dist.sample()
                    .detach()
                )
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(
                    dtype=weight_dtype
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[
                    min(placeholder_token_ids) : max(placeholder_token_ids) + 1
                ] = False

                with torch.no_grad():
                    accelerator.unwrap_model(
                        text_encoder
                    ).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[
                        index_no_updates
                    ]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.training.save_steps == 0:
                    save_path = os.path.join(
                        args.training.output_dir,
                        f"learned_embeds-steps-{global_step}.bin",
                    )
                    save_progress(
                        text_encoder,
                        placeholder_token_ids,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=True,
                    )

                if accelerator.is_main_process:
                    if global_step % args.training.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`

                        save_path = os.path.join(
                            args.training.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if (
                        args.validate.validation_prompt is not None
                        and global_step % args.validate.validation_steps == 0
                    ):
                        images = log_validation(
                            text_encoder,
                            tokenizer,
                            unet,
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            epoch,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.training.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.training.save_as_full_pipeline:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.training.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.training.output_dir)
        # Save the newly trained embeddings
        save_path = os.path.join(args.training.output_dir, "learned_embeds.bin")
        save_progress(
            text_encoder,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    import argparse

    passer = argparse.ArgumentParser()
    passer.add_argument("--config", help="Config to train textual inversion")
    args = parser.parse_args()
    main(config_file=args.config)
