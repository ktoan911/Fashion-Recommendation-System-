import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from databases.mongodb import ImageDB
from modules.clothes_detection import ClothesDetection
from modules.fashion_vlp import FashionVLP
from modules.landmark_detection import LandmarkDetection
from utils.logger import get_logger

logger = get_logger("EXTRACT_FTAR")


class FTarExtractor:
    def __init__(self, model_path="model/fashion_vlp.pt", device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")
        self.db = ImageDB()

        logger.info(f"Loading model from {model_path}...")
        self.model = FashionVLP().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                state_dict = {
                    key.replace("_orig_mod.", ""): value
                    for key, value in state_dict.items()
                }
            self.model.load_state_dict(state_dict)
        else:
            state_dict = checkpoint
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                state_dict = {
                    key.replace("_orig_mod.", ""): value
                    for key, value in state_dict.items()
                }
            self.model.load_state_dict(state_dict)

        self.model.eval()
        logger.info("Model loaded successfully!")

        self.clothes_detector = ClothesDetection()
        self.landmark_detector = LandmarkDetection()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_image(self, image_path):
        try:
            pil_image = Image.open(image_path).convert("RGB")

            cropped_image = self.clothes_detector.get_highest_confidence_object(
                pil_image
            )
            if cropped_image is None:
                cropped_image = pil_image
                logger.warning(
                    f"Could not detect clothing in {image_path}, using full image"
                )

            landmarks = self.landmark_detector.detect(pil_image)

            target_tensor = self.transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]
            cropped_tensor = self.transform(cropped_image).unsqueeze(
                0
            )  # [1, 3, 224, 224]

            landmark_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(
                0
            )  # [1, 14, 2]

            return target_tensor, cropped_tensor, landmark_tensor

        except Exception as e:
            logger.error(f"Error with {image_path}: {str(e)}")
            return None, None, None

    def extract_ftar(self, target_image, cropped_image, landmarks):
        with torch.no_grad():
            target_image = target_image.to(self.device)
            cropped_image = cropped_image.to(self.device)
            landmarks = landmarks.to(self.device)

            f_tar = self.model.target_block(target_image, cropped_image, landmarks)

            f_tar_np = f_tar.cpu().numpy().squeeze()  # [768]

            return f_tar_np

    def extract_from_folder(self, folder_path, image_extensions=None):
        if image_extensions is None:
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]

        # Tìm tất cả file ảnh
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))

        logger.info(f"Tìm thấy {len(image_files)} ảnh")

        if len(image_files) == 0:
            logger.warning("Không tìm thấy ảnh nào trong folder!")
            return

        features_dict = {}

        for image_path in tqdm(image_files, desc="Đang trích xuất đặc trưng"):
            # Preprocess ảnh
            target_img, cropped_img, landmarks = self.preprocess_image(image_path)

            if target_img is None:
                logger.error(f"Error: {image_path}")
                continue

            try:
                f_tar = self.extract_ftar(target_img, cropped_img, landmarks)
                relative_path = os.path.relpath(image_path, folder_path)
                features_dict[relative_path] = f_tar.tolist()

            except Exception as e:
                logger.error(f"Error extracting features for {image_path}: {str(e)}")
                continue

        self.db.insert_many(
            [{"name": p, "features": fs} for p, fs in features_dict.items()]
        )

        logger.info(f"Completed! Saved features for {len(features_dict)}")


def main():
    """
    Hàm main để chạy script
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract f_tar features from image folder"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/fashion_vlp.pt",
        help="Path to the model file (default: model/fashion_vlp.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run the model on (default: auto detect)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.folder_path):
        logger.error(f"Folder not found: {args.folder_path}")
        return

    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return

    device = torch.device(args.device) if args.device else None
    extractor = FTarExtractor(model_path=args.model_path, device=device)

    extractor.extract_from_folder(args.folder_path)


if __name__ == "__main__":
    main()
