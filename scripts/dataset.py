from torch.utils.data import Dataset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import PIL_INTERPOLATION
import cv2
from torchvision import transforms
import torch
from PIL import Image
import random
import numpy as np


template = ["A dog{}"]

class ConceptDataset(Dataset):

    def __init__(self,
                data_root,
                tokenizer,
                width=720,
                height=480,
                repeats=100,
                interpolation="bicubic",
                flip_p=0.5,
                split="train",
                placeholder_token="*",
                center_crop=False,
                original_prompt="A dog walking"):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.original_prompt=original_prompt

        self.videos_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_videos = len(self.videos_paths)
        self._length = self.num_videos

        if split == "train":
            self._length = self.num_videos * repeats 

        self.interpolation = {"linear": PIL_INTERPOLATION["linear"],
                                "bilinear": PIL_INTERPOLATION["bilinear"],
                                "bicubic": PIL_INTERPOLATION["bicubic"],
                                "lanczos": PIL_INTERPOLATION["lanczos"],}[interpolation]

        self.templates = template
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        
        video_path = self.videos_paths[i % self.num_videos]
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not read frames from video {video_path}")
        
        # Convert frames to tensor
        processed_frames = []
        for frame in frames:
            # Convert to PIL for consistent processing
            image = Image.fromarray(frame)
            
            # Process each frame as before
            img = np.array(image).astype(np.uint8)

            if self.center_crop: 
                raise NotImplementedError("Center cropping is not implemented yet for video.") 

            image = Image.fromarray(img)
            image = image.resize((self.width, self.height), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32) # Normalize to [-1, 1]
            processed_frames.append(image)

        # Stack frames into a single tensor
        video_tensor = torch.from_numpy(np.stack(processed_frames)).permute(3, 0, 1, 2) # TO [C, F, H, W]

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(text,
                                                padding="max_length",
                                                truncation=True,
                                                max_length=self.tokenizer.model_max_length, 
                                                return_tensors="pt",).input_ids[0]
        example["pixel_values"] = video_tensor
        example['original_prompt'] = self.tokenizer(self.original_prompt,
                                                padding="max_length",
                                                truncation=True,
                                                max_length=self.tokenizer.model_max_length, 
                                                return_tensors="pt",).input_ids[0]

        return example