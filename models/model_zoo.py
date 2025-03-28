# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
# limitations under the License.

"""Model definition."""
# pylint: disable=g-multiple-import,g-importing-member,g-bad-import-order,missing-class-docstring

from typing import Optional
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from models.languagebindmodel.languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer


class ModelZoo:

  def transform(self, image):
    pass

  def transform_tensor(self, image_tensor):
    pass

  def calculate_loss(
      self, output, target_images
  ):
    pass

  def get_probability(
      self, output, target_images
  ):
    pass


class LBsimilarity(ModelZoo):

  def __init__(self):

    self.clip_type = {'video': 'LanguageBind_Video_FT'}
    self.language_bind_model = LanguageBind(clip_type=self.clip_type, cache_dir='/home/joberant/NLP_2425a/baralon1/cache').to("cuda")
    self.language_bind_model.eval()
    self.modality_transform = {c: transform_dict[c](self.language_bind_model.modality_config[c]) for c in self.clip_type.keys()}

# TODO: Check if needed
  def transform_tensor(self, image_tensor):
    image_tensor = torch.nn.functional.interpolate(
        image_tensor, size=(224, 224), mode="bicubic", align_corners=False
    )
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    image_tensor = normalize(image_tensor)
    return image_tensor

# TODO: Check if needed
  def calculate_loss(self, output, target_images):
    # calculate CLIP loss
    output = self.clip_model.get_image_features(output)
    # loss = -torch.cosine_similarity(output, input_clip_embedding, axis=1)

    mean_target_image = target_images.mean(dim=0).reshape(1, -1)
    loss = torch.mean(
        torch.cosine_similarity(
            output[None], mean_target_image[:, None], axis=2
        ),
        axis=1,
    )
    loss = 1 - loss.mean()
    return loss

  def get_probability(self, video, target_videos):

    inputs = {'video': to_device(self.modality_transform['video'](video), "cuda")}
    with torch.no_grad():
        output = self.language_bind_model(inputs)['video']
        output /= output.norm(dim=-1, keepdim=True)

    mean_target_image = target_videos.mean(dim=0).reshape(1, -1) 
    loss = torch.mean(torch.cosine_similarity(output, mean_target_image, axis=1), axis=0)
    return loss.mean()
