import json
import os
import pickle
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from modules.clothes_detection import ClothesDetection
from modules.landmark_detection import LandmarkDetection
from utils.logger import get_logger

logger = get_logger("DATASET")


class FashionDataset(Dataset):
    def __init__(
        self,
        annotations_folder,
        folder_img,
        transform=None,
        type="train",
        cache_dir="cache",
    ):
        self.annotations_file = os.path.join(annotations_folder, f"{type}.json")
        self.annotations = self._load_annotations(self.annotations_file)
        self.folder_img = folder_img
        self.transform = transform
        self.type = type

        self.cache_dir = Path(cache_dir) / type
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.clothing_detection = None
        self.landmark_detection = None

        self.cache_file = self.cache_dir / "preprocessing_cache.pkl"
        if not self.cache_file.exists():
            logger.info("Cache not found. Create cache...")
            self._create_cache()
        else:
            logger.info("Loading cache...")
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Handle both old and new cache format
            if isinstance(cache_data, dict) and "sample_cache" in cache_data:
                self.cache = cache_data["sample_cache"]
                self.image_cache = cache_data["image_cache"]
                stats = cache_data.get("stats", {})
                logger.info(
                    f"✅ Cache loaded: {len(self.cache)} samples, {len(self.image_cache)} unique images"
                )
                if "duplicates_skipped" in stats:
                    logger.info(
                        f"   ⚡ Deduplication saved {stats['duplicates_skipped']} processing operations"
                    )
            else:
                # Old format - direct cache
                self.cache = cache_data
                self.image_cache = {}
                logger.info(f"📁 Legacy cache loaded: {len(self.cache)} samples")
                logger.info("   💡 Recreate cache to enable deduplication optimization")

    def _load_annotations(self, annotations_file):
        with open(annotations_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _create_cache(self):
      
        self.clothing_detection = ClothesDetection()
        self.landmark_detection = LandmarkDetection()

        self.cache = {}
        self.image_cache = {}  # filename -> processed results

        total_samples = len(self.annotations)
        processed_images = set()
        duplicate_count = 0

        logger.info(f"Creating cache for {total_samples} samples...")

        for idx in range(total_samples):
            if idx % 10000 == 0:
                unique_images = len(processed_images)
                logger.info(
                    f"Progress: {idx}/{total_samples} | Unique images: {unique_images} | Duplicates skipped: {duplicate_count}"
                )

            ref_filename = self.annotations[idx]["candidate"] + ".jpg"
            target_filename = self.annotations[idx]["target"] + ".jpg"

            ref_img_path = os.path.join(self.folder_img, ref_filename)
            target_img_path = os.path.join(self.folder_img, target_filename)

            if ref_filename not in self.image_cache:
                ref_image = Image.open(ref_img_path).convert("RGB")
                crop_ref_image = self.clothing_detection.get_highest_confidence_object(
                    ref_image
                )

                # Detect landmarks
                if crop_ref_image is not None:
                    landmarks = self.landmark_detection.detect(crop_ref_image)
                else:
                    landmarks = self.landmark_detection.detect(ref_image)

                ref_result = {
                    "crop_available": crop_ref_image is not None,
                    "landmarks": landmarks,
                    "image_size": ref_image.size,
                    "crop_path": None,
                }

                if crop_ref_image is not None:
                    crop_path = self.cache_dir / f"crop_{ref_filename}"
                    crop_ref_image.save(crop_path)
                    ref_result["crop_path"] = str(crop_path)

                self.image_cache[ref_filename] = ref_result
                processed_images.add(ref_filename)
            else:
                if self.image_cache[ref_filename]["landmarks"] is None:
                    ref_image = Image.open(ref_img_path).convert("RGB")
                    crop_ref_image = (
                        self.clothing_detection.get_highest_confidence_object(ref_image)
                    )

                    # Detect landmarks for reference
                    if crop_ref_image is not None:
                        landmarks = self.landmark_detection.detect(crop_ref_image)
                    else:
                        landmarks = self.landmark_detection.detect(ref_image)

                    self.image_cache[ref_filename]["landmarks"] = landmarks

                duplicate_count += 1

            if target_filename not in self.image_cache:
                target_image = Image.open(target_img_path).convert("RGB")
                crop_target_image = (
                    self.clothing_detection.get_highest_confidence_object(target_image)
                )
                target_result = {
                    "crop_available": crop_target_image is not None,
                    "landmarks": None,  # Not used for target images
                    "image_size": target_image.size,
                    "crop_path": None,
                }

                if crop_target_image is not None:
                    crop_path = self.cache_dir / f"crop_{target_filename}"
                    crop_target_image.save(crop_path)
                    target_result["crop_path"] = str(crop_path)

                self.image_cache[target_filename] = target_result
                processed_images.add(target_filename)
            else:
                duplicate_count += 1

            self.cache[idx] = {
                "ref_filename": ref_filename,
                "target_filename": target_filename,
                "crop_ref_available": self.image_cache[ref_filename]["crop_available"],
                "crop_target_available": self.image_cache[target_filename][
                    "crop_available"
                ],
                "landmarks": self.image_cache[ref_filename]["landmarks"],
                "ref_image_size": self.image_cache[ref_filename]["image_size"],
                "target_image_size": self.image_cache[target_filename]["image_size"],
                "crop_ref_path": self.image_cache[ref_filename]["crop_path"],
                "crop_target_path": self.image_cache[target_filename]["crop_path"],
            }

        # Save both caches to disk
        cache_data = {
            "sample_cache": self.cache,
            "image_cache": self.image_cache,
            "stats": {
                "total_samples": total_samples,
                "unique_images": len(processed_images),
                "duplicates_skipped": duplicate_count,
                "compression_ratio": len(processed_images)
                / (total_samples * 2),  # 2 images per sample
            },
        }

        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        logger.info("\n✅ Cache created successfully:")
        logger.info(f"   📊 Total samples: {total_samples}")
        logger.info(f"Disk space saved: ~{duplicate_count * 0.1:.1f}MB")

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

        # Load cropped images from cache (using unique filenames)
        if cached_data["crop_ref_available"] and cached_data["crop_ref_path"]:
            crop_ref_image = Image.open(cached_data["crop_ref_path"]).convert("RGB")
        else:
            crop_ref_image = ref_image

        if cached_data["crop_target_available"] and cached_data["crop_target_path"]:
            crop_target_image = Image.open(cached_data["crop_target_path"]).convert(
                "RGB"
            )
        else:
            crop_target_image = target_image

        landmarks = cached_data.get("landmarks", None)

        if self.transform:
            ref_image_resized = self.transform(ref_image)
            crop_ref_image = self.transform(crop_ref_image)
            target_image_resized = self.transform(target_image)
            crop_target_image = self.transform(crop_target_image)

            ref_image_size = cached_data.get("ref_image_size", ref_image.size)
            landmarks = self.resize_points_from_cache(ref_image_size, landmarks)

        sample = {
            "reference_image": ref_image_resized,
            "crop_reference_image": crop_ref_image,
            "target_image": target_image_resized,
            "crop_target_image": crop_target_image,
            "feedback_tokens": torch.tensor(feedback_tokens),
            "landmarks": landmarks,
        }
        return sample

    def resize_points_from_cache(self, original_size, points, new_size=(224, 224)):

        if points is None or len(points) == 0:
            return torch.zeros(14, 2, dtype=torch.float32)

        W_orig, H_orig = original_size
        W_new, H_new = new_size

        scaled_points = [
            [x * (W_new / W_orig), y * (H_new / H_orig)] for (x, y) in points
        ]
        return torch.tensor(scaled_points, dtype=torch.float32)
