import json
import os
import pickle
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from modules.clothes_detection import ClothesDetection
from modules.landmark_detection import LandmarkDetection


class CachedFashionDataset(Dataset):
    def __init__(self, annotations_folder, folder_img, transform=None, type="train", cache_dir="cache"):
        self.annotations_file = os.path.join(annotations_folder, f"{type}.json")
        self.annotations = self._load_annotations(self.annotations_file)
        self.folder_img = folder_img
        self.transform = transform
        self.type = type
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir) / type
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detection modules only if needed
        self.clothing_detection = None
        self.landmark_detection = None
        
        # Check if cache exists, if not create it
        self.cache_file = self.cache_dir / "preprocessing_cache.pkl"
        if not self.cache_file.exists():
            print(f"Cache không tồn tại cho {type} set. Đang tạo cache...")
            self._create_cache()
        else:
            print(f"Đang load cache cho {type} set...")
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"Đã load cache thành công cho {len(self.cache)} samples")

    def _load_annotations(self, annotations_file):
        with open(annotations_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _create_cache(self):
        """Tạo cache cho toàn bộ dataset để tránh tính toán lại clothes detection và landmark detection"""
        self.clothing_detection = ClothesDetection()
        self.landmark_detection = LandmarkDetection()
        
        self.cache = {}
        total_samples = len(self.annotations)
        
        for idx in range(total_samples):
            if idx % 100 == 0:
                print(f"Caching progress: {idx}/{total_samples}")
                
            ref_img_path = os.path.join(
                self.folder_img, self.annotations[idx]["candidate"] + ".jpg"
            )
            target_img_path = os.path.join(
                self.folder_img, self.annotations[idx]["target"] + ".jpg"
            )

            # Load images
            ref_image = Image.open(ref_img_path).convert("RGB")
            target_image = Image.open(target_img_path).convert("RGB")
            
            # Perform clothes detection
            crop_ref_image = self.clothing_detection.get_highest_confidence_object(ref_image)
            crop_target_image = self.clothing_detection.get_highest_confidence_object(target_image)
            
            # Perform landmark detection
            if crop_ref_image is not None:
                landmarks = self.landmark_detection.detect(crop_ref_image)
            else:
                landmarks = self.landmark_detection.detect(ref_image)
            
            # Store in cache
            self.cache[idx] = {
                'crop_ref_available': crop_ref_image is not None,
                'crop_target_available': crop_target_image is not None,
                'landmarks': landmarks,
                'ref_image_size': ref_image.size,
                'target_image_size': target_image.size
            }
            
            # Store cropped images if available
            if crop_ref_image is not None:
                crop_ref_path = self.cache_dir / f"crop_ref_{idx}.jpg"
                crop_ref_image.save(crop_ref_path)
                self.cache[idx]['crop_ref_path'] = str(crop_ref_path)
            
            if crop_target_image is not None:
                crop_target_path = self.cache_dir / f"crop_target_{idx}.jpg"
                crop_target_image.save(crop_target_path)
                self.cache[idx]['crop_target_path'] = str(crop_target_path)
        
        # Save cache to disk
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        
        print(f"Cache đã được tạo và lưu cho {total_samples} samples")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load original images
        ref_img_path = os.path.join(
            self.folder_img, self.annotations[idx]["candidate"] + ".jpg"
        )
        target_img_path = os.path.join(
            self.folder_img, self.annotations[idx]["target"] + ".jpg"
        )
        feedback_tokens = self.annotations[idx]["feedback_tokens"]

        ref_image = Image.open(ref_img_path).convert("RGB")
        target_image = Image.open(target_img_path).convert("RGB")
        
        # Get cached data
        cached_data = self.cache[idx]
        
        # Load cropped images from cache
        if cached_data['crop_ref_available']:
            crop_ref_image = Image.open(cached_data['crop_ref_path']).convert("RGB")
        else:
            crop_ref_image = ref_image
            
        if cached_data['crop_target_available']:
            crop_target_image = Image.open(cached_data['crop_target_path']).convert("RGB")
        else:
            crop_target_image = target_image
        
        # Get cached landmarks
        landmarks = cached_data['landmarks']

        # Apply transforms
        if self.transform:
            ref_image_resized = self.transform(ref_image)
            crop_ref_image = self.transform(crop_ref_image)
            target_image_resized = self.transform(target_image)
            crop_target_image = self.transform(crop_target_image)
            
            # Resize landmarks based on original image size
            landmarks = self.resize_points_from_cache(
                cached_data['ref_image_size'], landmarks
            )

        sample = {
            "reference_image": ref_image_resized,
            "crop_reference_image": crop_ref_image,
            "target_image": target_image_resized,
            "crop_target_image": crop_target_image,
            "feedback_tokens": torch.tensor(feedback_tokens),
            "landmarks": landmarks
        }
        return sample

    def resize_points_from_cache(self, original_size, points, new_size=(224, 224)):
        """
        Resize landmarks từ original size về new_size
        """
        W_orig, H_orig = original_size
        W_new, H_new = new_size

        scaled_points = [
            [x * (W_new / W_orig), y * (H_new / H_orig)] for (x, y) in points
        ]
        return torch.tensor(scaled_points, dtype=torch.float32)