import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from modules.clothes_detection import ClothesDetection
from modules.landmark_detection import LandmarkDetection


class FashionDataset(Dataset):
    def __init__(self, annotations_folder, folder_img, transform=None, type="train"):
        annotations_file = os.path.join(annotations_folder, f"{type}.json")
        self.annotations = self._load_annotations(annotations_file)
        self.folder_img = folder_img
        self.transform = transform
        self.clothing_detection = ClothesDetection()
        self.landmark_detection = LandmarkDetection()

    def _load_annotations(self, annotations_file):
        with open(annotations_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ref_img_path = os.path.join(
            self.folder_img, self.annotations[idx]["candidate"] + ".jpg"
        )
        target_img_path = os.path.join(
            self.folder_img, self.annotations[idx]["target"] + ".jpg"
        )
        feedback_tokens = self.annotations[idx]["feedback_tokens"]

        ref_image = Image.open(ref_img_path).convert("RGB")
        crop_ref_image = self.clothing_detection.get_highest_confidence_object(
            ref_image
        )
        target_image = Image.open(target_img_path).convert("RGB")
        crop_target_image = self.clothing_detection.get_highest_confidence_object(
            target_image
        )
        landmarks = (
            self.landmark_detection.detect(crop_ref_image)
            if crop_ref_image
            else self.landmark_detection.detect(ref_image)
        )

        if self.transform:
            ref_image_resized = self.transform(ref_image)
            crop_ref_image = (
                self.transform(crop_ref_image) if crop_ref_image else crop_ref_image
            )
            target_image_resized = self.transform(target_image)
            crop_target_image = (
                self.transform(crop_target_image)
                if crop_target_image
                else crop_target_image
            )
            landmarks = self.resize_points(ref_image, landmarks)

        sample = {
            "reference_image": ref_image_resized,
            "crop_reference_image": crop_ref_image,
            "target_image": target_image_resized,
            "crop_target_image": crop_target_image,
            "feedback_tokens": torch.tensor(feedback_tokens),
            "landmarks": landmarks
            if landmarks.numel() != 0
            else torch.zeros((1, 2), dtype=torch.float32),
        }
        return sample

    def resize_points(self, pil_img, points, new_size=(224, 224)):
        """
        pil_img : ảnh PIL Image (RGB)
        points  : list các điểm [(x1, y1), (x2, y2)] dạng pixel
        new_size: kích thước mới (width, height)
        """
        W_orig, H_orig = pil_img.size
        W_new, H_new = new_size

        scaled_points = [
            [x * (W_new / W_orig), y * (H_new / H_orig)] for (x, y) in points
        ]
        return torch.tensor(scaled_points, dtype=torch.float32)
