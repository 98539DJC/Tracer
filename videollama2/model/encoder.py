import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from transformers import (
    CLIPVisionModel,   CLIPImageProcessor,   CLIPVisionConfig,
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig,
)


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, load_pretrained=False):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        config._attn_implementation = "flash_attention_2"

        if not load_pretrained:
            self.vision_tower = CLIPVisionModel(config=config)
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionTower(nn.Module):

    def __init__(self, vision_tower, args, load_pretrained=False):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        config = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
        config._attn_implementation = 'eager'

        if not load_pretrained:
            self.vision_tower = SiglipVisionModel(config=config)
        else:
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class PrecomputedFeatureTower(nn.Module):
    """Zero-weight stand-in for the vision tower when features are precomputed.

    SigLIP costs ~2.6 GB of VRAM (weights + activations) plus a forward pass over
    every sampled frame.  Once the per-video features have been cached offline
    (tools/precompute_msvd_siglip_features.py) the tower is never needed, so we
    keep only the metadata the rest of the pipeline reads off it:
    hidden_size / image_size / num_patches / image_processor.
    """

    DEFAULT_NAME = 'google/siglip-so400m-patch14-384'

    def __init__(self, args, vision_tower_name=None, hidden_size=1152,
                 image_size=384, patch_size=14, **kwargs):
        super().__init__()
        self.vision_tower_name = vision_tower_name or self.DEFAULT_NAME
        self._hidden_size = int(hidden_size)
        self._image_size = int(image_size)
        self._patch_size = int(patch_size)
        # The processor holds no weights; keep it so process_image /
        # process_video still work for any non-video inputs.
        self.image_processor = None
        try:
            self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        except Exception:
            pass
        self.config = SimpleNamespace(
            hidden_size=self._hidden_size,
            image_size=self._image_size,
            patch_size=self._patch_size,
        )

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def num_patches(self):
        return (self._image_size // self._patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self._image_size // self._patch_size

    @property
    def dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            'PrecomputedFeatureTower.forward() was called although the vision tower '
            'is disabled (--use_precomputed_features). Pass --video_feature_folder '
            'so the dataset yields cached features instead of raw pixels.'
        )


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower is not None and 'precomputed' in str(vision_tower).lower():
        vision_tower = PrecomputedFeatureTower(vision_tower_cfg, **kwargs)
    elif 'clip' in vision_tower:
        vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower:
        vision_tower = SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')

    return vision_tower
