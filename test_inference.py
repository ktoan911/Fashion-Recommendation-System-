#!/usr/bin/env python3
"""
Usage:
    python inference.py --ref_image path/to/reference.jpg --target_image path/to/target.jpg --text "more colorful"

hoặc

    python inference.py --batch_inference --image_folder path/to/images --text "more casual"
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.append("/media/DATA/Fashion-Recommendation-System-")

from modules.clothes_detection import ClothesDetection
from modules.fashion_vlp import FashionVLP
from modules.landmark_detection import LandmarkDetection
from utils.data_utils import tokenize_text


class FashionVLPInference:
    """
    Class để thực hiện inference với Fashion VLP model
    """

    def __init__(
        self,
        model_path: str = "/media/DATA/Fashion-Recommendation-System-/model/fashion_vlp.pt",
    ):
        """
        Khởi tạo inference engine

        Args:
            model_path: Đường dẫn đến file model .pt
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng device: {self.device}")

        # Load model
        self.model = self._load_model(model_path)

        # Initialize preprocessing modules
        self.clothes_detection = ClothesDetection(
            "/media/DATA/Fashion-Recommendation-System-/model/yolov8n_finetune.pt"
        )
        self.landmark_detection = LandmarkDetection()

        # Image transform pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        print("Fashion VLP Inference Engine đã sẵn sàng!")

    def _load_model(self, model_path: str) -> FashionVLP:
        """Load model từ checkpoint"""
        print(f"Đang load model từ: {model_path}")

        # Khởi tạo model với config mặc định
        model = FashionVLP(
            d_model=768,
            resnet_model_name="resnet50",
            vlp_model_name="bert-base-uncased",
        )

        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Xử lý các định dạng checkpoint khác nhau
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(
                    f"Loaded model từ checkpoint epoch: {checkpoint.get('epoch', 'unknown')}"
                )
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

        except Exception as e:
            print(f"Lỗi khi load model: {e}")
            print("Đang thử load trực tiếp state dict...")
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e2:
                print(f"Không thể load model: {e2}")
                raise

        model.to(self.device)
        model.eval()
        print("Model đã được load thành công!")
        return model

    def _preprocess_image(
        self, image_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        Tiền xử lý ảnh: resize, crop quần áo, detect landmarks

        Args:
            image_path: Đường dẫn đến ảnh

        Returns:
            Tuple (original_image_tensor, cropped_image_tensor, landmarks)
        """
        # Load và convert ảnh
        pil_image = Image.open(image_path).convert("RGB")

        # Detect và crop quần áo
        cropped_image = self.clothes_detection.get_highest_confidence_object(pil_image)
        if cropped_image is None:
            print(
                f"Cảnh báo: Không detect được quần áo trong {image_path}, sử dụng ảnh gốc"
            )
            cropped_image = pil_image

        # Detect landmarks
        landmarks = self.landmark_detection.detect(
            cropped_image if cropped_image else pil_image
        )

        # Transform images
        original_tensor = self.transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]
        cropped_tensor = self.transform(cropped_image).unsqueeze(0)  # [1, 3, 224, 224]

        # Convert landmarks to tensor và resize để phù hợp với ảnh 224x224
        landmarks_resized = self._resize_landmarks(
            pil_image.size, landmarks, (224, 224)
        )

        return (
            original_tensor.to(self.device),
            cropped_tensor.to(self.device),
            landmarks_resized,
        )

    def _resize_landmarks(
        self,
        original_size: Tuple[int, int],
        landmarks: List[List[float]],
        new_size: Tuple[int, int],
    ) -> List[List[float]]:
        """Resize landmarks tương ứng với ảnh đã resize"""
        if not landmarks:
            return [[0, 0] for _ in range(14)]  # Default empty landmarks

        old_w, old_h = original_size
        new_w, new_h = new_size

        resized_landmarks = []
        for point in landmarks:
            if len(point) >= 2:
                # Landmarks từ mediapipe đã được normalize [0,1], chỉ cần giữ nguyên
                resized_landmarks.append([point[0], point[1]])
            else:
                resized_landmarks.append([0, 0])

        # Đảm bảo có đủ 14 landmarks
        while len(resized_landmarks) < 14:
            resized_landmarks.append([0, 0])

        return resized_landmarks[:14]

    def _preprocess_text(self, text: str) -> torch.Tensor:
        """Tokenize text feedback"""
        tokens = tokenize_text(text)
        return torch.tensor(tokens).unsqueeze(0).to(self.device)  # [1, seq_len]

    def compute_similarity(
        self, ref_image_path: str, target_image_path: str, text_feedback: str
    ) -> float:
        """
        Tính toán độ tương đồng giữa reference image + text và target image

        Args:
            ref_image_path: Đường dẫn đến ảnh reference
            target_image_path: Đường dẫn đến ảnh target
            text_feedback: Text mô tả sự thay đổi mong muốn

        Returns:
            Cosine similarity score (0-1)
        """
        with torch.no_grad():
            # Preprocess inputs
            ref_img, ref_crop, ref_landmarks = self._preprocess_image(ref_image_path)
            target_img, target_crop, target_landmarks = self._preprocess_image(
                target_image_path
            )
            text_tokens = self._preprocess_text(text_feedback)

            # Convert landmarks to tensor
            ref_landmarks_tensor = torch.tensor(ref_landmarks, dtype=torch.float32).to(
                self.device
            )
            target_landmarks_tensor = torch.tensor(
                target_landmarks, dtype=torch.float32
            ).to(self.device)

            # Forward pass
            f_ref, f_tar = self.model(
                ref_image=ref_img,
                text_tokens=text_tokens,
                target_image=target_img,
                crop_reference_image=ref_crop,
                crop_target_image=target_crop,
                landmark_locations=ref_landmarks_tensor.unsqueeze(0),  # [1, 14, 2]
            )

            # Compute cosine similarity
            f_ref_norm = F.normalize(f_ref, p=2, dim=1)
            f_tar_norm = F.normalize(f_tar, p=2, dim=1)
            similarity = torch.mm(f_ref_norm, f_tar_norm.t()).item()

            return similarity

    def find_best_match(
        self,
        ref_image_path: str,
        text_feedback: str,
        candidate_images: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Tìm những ảnh target phù hợp nhất với reference + text feedback

        Args:
            ref_image_path: Đường dẫn ảnh reference
            text_feedback: Text mô tả mong muốn
            candidate_images: List đường dẫn các ảnh candidate
            top_k: Số lượng kết quả trả về

        Returns:
            List các tuple (image_path, similarity_score) được sắp xếp theo score giảm dần
        """
        results = []

        print(f"Đang tính toán similarity cho {len(candidate_images)} ảnh...")

        for i, candidate_path in enumerate(candidate_images):
            try:
                similarity = self.compute_similarity(
                    ref_image_path, candidate_path, text_feedback
                )
                results.append((candidate_path, similarity))

                if (i + 1) % 10 == 0:
                    print(f"Đã xử lý {i + 1}/{len(candidate_images)} ảnh")

            except Exception as e:
                print(f"Lỗi khi xử lý {candidate_path}: {e}")
                continue

        # Sắp xếp theo similarity giảm dần
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def batch_inference_folder(
        self, ref_image_path: str, text_feedback: str, image_folder: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Thực hiện inference trên toàn bộ ảnh trong folder
        """
        # Lấy tất cả ảnh trong folder
        supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        candidate_images = []

        for filename in os.listdir(image_folder):
            if any(filename.lower().endswith(ext) for ext in supported_formats):
                candidate_images.append(os.path.join(image_folder, filename))

        print(f"Tìm thấy {len(candidate_images)} ảnh trong folder {image_folder}")

        return self.find_best_match(
            ref_image_path, text_feedback, candidate_images, top_k
        )


def main():
    parser = argparse.ArgumentParser(description="Fashion VLP Model Inference")
    parser.add_argument(
        "--model_path",
        default="/media/DATA/Fashion-Recommendation-System-/model/fashion_vlp.pt",
        help="Đường dẫn đến file model",
    )
    parser.add_argument("--ref_image", help="Đường dẫn đến ảnh reference")
    parser.add_argument(
        "--target_image", help="Đường dẫn đến ảnh target (cho single inference)"
    )
    parser.add_argument(
        "--text", required=True, help="Text feedback mô tả thay đổi mong muốn"
    )
    parser.add_argument(
        "--batch_inference", action="store_true", help="Thực hiện batch inference"
    )
    parser.add_argument(
        "--image_folder", help="Folder chứa ảnh candidates (cho batch inference)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Số lượng kết quả top trả về"
    )
    parser.add_argument("--output", help="File để lưu kết quả (format JSON)")

    args = parser.parse_args()

    if not args.ref_image:
        print("Lỗi: Cần cung cấp --ref_image")
        return

    # Khởi tạo inference engine
    engine = FashionVLPInference(args.model_path)

    if args.batch_inference:
        if not args.image_folder:
            print("Lỗi: Cần cung cấp --image_folder cho batch inference")
            return

        print("\n=== BATCH INFERENCE ===")
        print(f"Reference image: {args.ref_image}")
        print(f"Text feedback: '{args.text}'")
        print(f"Searching in folder: {args.image_folder}")

        results = engine.batch_inference_folder(
            args.ref_image, args.text, args.image_folder, args.top_k
        )

        print(f"\n=== TOP {len(results)} KẾT QUẢ ===")
        for i, (image_path, score) in enumerate(results, 1):
            print(f"{i}. {os.path.basename(image_path)} - Score: {score:.4f}")

        # Lưu kết quả nếu có output file
        if args.output:
            result_data = {
                "reference_image": args.ref_image,
                "text_feedback": args.text,
                "results": [
                    {"image_path": path, "similarity_score": float(score)}
                    for path, score in results
                ],
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"\nKết quả đã được lưu vào: {args.output}")

    else:
        if not args.target_image:
            print("Lỗi: Cần cung cấp --target_image cho single inference")
            return

        print("\n=== SINGLE INFERENCE ===")
        print(f"Reference image: {args.ref_image}")
        print(f"Target image: {args.target_image}")
        print(f"Text feedback: '{args.text}'")

        similarity = engine.compute_similarity(
            args.ref_image, args.target_image, args.text
        )

        print("\n=== KẾT QUẢ ===")
        print(f"Cosine Similarity: {similarity:.4f}")

        if similarity > 0.7:
            print("✅ Độ tương đồng cao - Ảnh target phù hợp với yêu cầu")
        elif similarity > 0.5:
            print("⚠️  Độ tương đồng trung bình - Ảnh target có thể phù hợp")
        else:
            print("❌ Độ tương đồng thấp - Ảnh target không phù hợp với yêu cầu")


if __name__ == "__main__":
    main()
