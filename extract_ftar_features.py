import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from datasets.mongodb import ImageDB
from modules.clothes_detection import ClothesDetection
from modules.fashion_vlp import FashionVLP
from modules.landmark_detection import LandmarkDetection
from utils.logger import get_logger

logger = get_logger("EXTRACT_FTAR")


class FTarExtractor:
    def __init__(self, model_path="model/fashion_vlp.pt", device=None):
        """
        Khởi tạo extractor để trích xuất đặc trưng f_tar từ ảnh

        Args:
            model_path (str): Đường dẫn đến file model đã train
            device: Device để chạy model (cuda hoặc cpu)
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Sử dụng device: {self.device}")
        self.db = ImageDB()

        # Load model
        logger.info(f"Đang load model từ {model_path}...")
        self.model = FashionVLP().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Xử lý trường hợp model được compile với torch.compile()
            # Loại bỏ prefix "_orig_mod." nếu có
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            state_dict = checkpoint
            # Xử lý trường hợp model được compile với torch.compile()
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict)

        self.model.eval()
        logger.info("Model đã được load thành công!")

        # Khởi tạo các module phụ trợ
        self.clothes_detector = ClothesDetection()
        self.landmark_detector = LandmarkDetection()

        # Transform cho ảnh
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
        """
        Tiền xử lý ảnh: load, detect clothes, detect landmarks

        Args:
            image_path (str): Đường dẫn đến ảnh

        Returns:
            tuple: (target_image_tensor, cropped_image_tensor, landmark_tensor)
        """
        try:
            # Load ảnh
            pil_image = Image.open(image_path).convert("RGB")

            # Detect và crop quần áo
            cropped_image = self.clothes_detector.get_highest_confidence_object(
                pil_image
            )
            if cropped_image is None:
                # Nếu không detect được, dùng toàn bộ ảnh
                cropped_image = pil_image
                logger.warning(
                    f"Không detect được quần áo trong {image_path}, sử dụng toàn bộ ảnh"
                )

            # Detect landmarks
            landmarks = self.landmark_detector.detect(pil_image)

            # Transform ảnh
            target_tensor = self.transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]
            cropped_tensor = self.transform(cropped_image).unsqueeze(
                0
            )  # [1, 3, 224, 224]

            # Convert landmarks thành tensor
            landmark_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(
                0
            )  # [1, 14, 2]

            return target_tensor, cropped_tensor, landmark_tensor

        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            return None, None, None

    def extract_ftar(self, target_image, cropped_image, landmarks):
        """
        Trích xuất đặc trưng f_tar từ ảnh

        Args:
            target_image: Tensor ảnh gốc [1, 3, 224, 224]
            cropped_image: Tensor ảnh đã crop [1, 3, 224, 224]
            landmarks: Tensor landmarks [1, 14, 2]

        Returns:
            numpy.ndarray: Vector đặc trưng f_tar
        """
        with torch.no_grad():
            # Chuyển tensors lên device
            target_image = target_image.to(self.device)
            cropped_image = cropped_image.to(self.device)
            landmarks = landmarks.to(self.device)

            # Chỉ sử dụng target block để extract f_tar
            f_tar = self.model.target_block(target_image, cropped_image, landmarks)

            # Chuyển về CPU và convert thành numpy
            f_tar_np = f_tar.cpu().numpy().squeeze()  # [768]

            return f_tar_np

    def extract_from_folder(self, folder_path, image_extensions=None):
        """
        Trích xuất f_tar từ tất cả ảnh trong folder và lưu vào JSON

        Args:
            folder_path (str): Đường dẫn đến folder chứa ảnh
            output_json_path (str): Đường dẫn file JSON output
            image_extensions (list): Danh sách extension ảnh cần xử lý
        """
        if image_extensions is None:
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]

        logger.info(f"Đang tìm ảnh trong folder: {folder_path}")

        # Tìm tất cả file ảnh
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))

        logger.info(f"Tìm thấy {len(image_files)} ảnh")

        if len(image_files) == 0:
            logger.warning("Không tìm thấy ảnh nào trong folder!")
            return

        # Dictionary để lưu kết quả
        features_dict = {}

        # Xử lý từng ảnh
        for image_path in tqdm(image_files, desc="Đang trích xuất đặc trưng"):
            # Preprocess ảnh
            target_img, cropped_img, landmarks = self.preprocess_image(image_path)

            if target_img is None:
                logger.error(f"Không thể xử lý ảnh: {image_path}")
                continue

            try:
                # Extract f_tar
                f_tar = self.extract_ftar(target_img, cropped_img, landmarks)

                # Lưu vào dictionary với key là tên file
                relative_path = os.path.relpath(image_path, folder_path)
                features_dict[relative_path] = f_tar.tolist()

                logger.info(f"Đã trích xuất đặc trưng cho: {relative_path}")

            except Exception as e:
                logger.error(f"Lỗi khi trích xuất đặc trưng cho {image_path}: {str(e)}")
                continue

        self.db.insert_many(
            [{"name": p, "features": fs} for p, fs in features_dict.items()]
        )

        logger.info(f"Hoàn thành! Đã lưu đặc trưng của {len(features_dict)}")


def main():
    """
    Hàm main để chạy script
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Trích xuất đặc trưng f_tar từ folder ảnh"
    )
    parser.add_argument(
        "--folder_path", type=str, required=True, help="Đường dẫn đến folder chứa ảnh"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/fashion_vlp.pt",
        help="Đường dẫn đến model file (mặc định: model/fashion_vlp.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device để chạy model (mặc định: auto detect)",
    )

    args = parser.parse_args()

    # Kiểm tra folder tồn tại
    if not os.path.exists(args.folder_path):
        logger.error(f"Folder không tồn tại: {args.folder_path}")
        return

    # Kiểm tra model file tồn tại
    if not os.path.exists(args.model_path):
        logger.error(f"Model file không tồn tại: {args.model_path}")
        return

    # Khởi tạo extractor
    device = torch.device(args.device) if args.device else None
    extractor = FTarExtractor(model_path=args.model_path, device=device)

    # Trích xuất đặc trưng
    extractor.extract_from_folder(args.folder_path)


if __name__ == "__main__":
    main()
