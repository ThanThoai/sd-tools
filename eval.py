import os
import random
import time

import PIL
import torch
from torch.nn.functional import cosine_similarity
from torchvision import transforms
import numpy as np
import clip


class CLIPEvaluator:
    def __init__(self, device, clip_model="ViT-B/32") -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
        self.preprocess = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
            + clip_preprocess.transforms[:2]
            + clip_preprocess.transforms[4:]  # to match CLIP input scale assumptions
        )  # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def get_image_features(self, images) -> torch.Tensor:
        if isinstance(images[0], PIL.Image.Image):
            # images is a list of PIL Images
            images = torch.stack([self.clip_preprocess(image) for image in images]).to(
                self.device
            )
        else:
            # images is a tensor of [-1, 1] images
            images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens)
        return text_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)
        if src_img_features.shape[0] == gen_img_features.shape[0]:
            return cosine_similarity(src_img_features, gen_img_features).mean().item()
        else:
            scores = []
            for idx in range(src_img_features.shape[0]):
                src_img_feature = src_img_features[idx].unsqueeze(0)
                scores.append(
                    cosine_similarity(src_img_feature, gen_img_features).mean().item()
                )
            return np.mean(scores)

    def txt_to_img_similarity(self, text, generated_images, reduction=True):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        if reduction:
            return cosine_similarity(text_features, gen_img_features).mean().item()
        else:
            return cosine_similarity(text_features, gen_img_features)


def read_dir_image(path):
    images = []
    for img in os.listdir(path):
        images.append(PIL.Image.open(os.path.join(path, img)))
    return images


if __name__ == "__main__":
    clip_eval = CLIPEvaluator(device="cuda")
    src_images = read_dir_image("data")
    generate_images = read_dir_image("generate")
    print(
        clip_eval.img_to_img_similarity(
            src_images=src_images, generated_images=generate_images
        )
    )
    # text = "a photo of"
    # print(clip_eval.txt_to_img_similarity(text=text, generated_images=generate_images))
