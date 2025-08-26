from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from databases.mongodb import ImageDB
from modules.clothes_detection import ClothesDetection
from modules.fashion_vlp import FashionVLP
from modules.landmark_detection import LandmarkDetection
from utils.data_utils import tokenize_text


class FashionVLPInference:
    def __init__(
        self,
        model_path: str = "model/fashion_vlp.pt",
        size_image=(224, 224),
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng device: {self.device}")

        self.model = self._load_model(model_path)
        self.size_image = (224, 224)
        self.clothes_detection = ClothesDetection(
            "model/yolov8n_finetune.pt"
        )
        self.landmark_detection = LandmarkDetection()

        # Image transform pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size_image),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.db = ImageDB()

        print("Fashion VLP Inference Engine đã sẵn sàng!")

    def _load_model(self, model_path: str) -> FashionVLP:
        print(f"Đang load model từ: {model_path}")

        model = FashionVLP(
            d_model=768,
            resnet_model_name="resnet50",
            vlp_model_name="bert-base-uncased",
        )
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                    state_dict = {
                        key.replace("_orig_mod.", ""): value
                        for key, value in state_dict.items()
                    }
                model.load_state_dict(state_dict)

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
        pil_image = Image.open(image_path).convert("RGB")

        cropped_image = self.clothes_detection.get_highest_confidence_object(pil_image)
        if cropped_image is None:
            cropped_image = pil_image
        landmarks = self.landmark_detection.detect(
            cropped_image if cropped_image else pil_image
        )
        original_tensor = self.transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]
        cropped_tensor = self.transform(cropped_image).unsqueeze(0)  # [1, 3, 224, 224]

        landmarks_resized = self._resize_landmarks(
            pil_image.size, landmarks, self.size_image
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
        if not landmarks:
            return [[0, 0] for _ in range(14)]

        old_w, old_h = original_size
        new_w, new_h = new_size

        resized_landmarks = []
        for point in landmarks:
            if len(point) >= 2:
                resized_landmarks.append([point[0], point[1]])
            else:
                resized_landmarks.append([0, 0])

        while len(resized_landmarks) < 14:
            resized_landmarks.append([0, 0])

        return resized_landmarks[:14]

    def compute_similarity(
        self, ref_image_path: str, text_feedback: str, top_k: int = 10
    ) -> float:
        with torch.no_grad():
            # Preprocess inputs
            ref_img, ref_crop, ref_landmarks = self._preprocess_image(ref_image_path)
            text_tokens = tokenize_text(text_feedback)

            ref_img = ref_img.to(self.device)
            ref_crop = ref_crop.to(self.device)
            ref_landmarks = torch.tensor(
                ref_landmarks, device=self.device, dtype=torch.float32
            )
            if ref_landmarks.ndim == 2:
                ref_landmarks = ref_landmarks.unsqueeze(0)

            f_ref = self.model.reference_block(
                ref_img,
                ref_crop,
                text_tokens,
                ref_landmarks,
            ).tolist()

            sim = self.db.vector_search(f_ref[0], k=top_k)

            return sim
