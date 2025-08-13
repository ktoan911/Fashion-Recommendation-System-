import timm  # Thư viện hữu ích cho các backbone CNN
import torch.nn as nn
from blocks import ReferenceBlock, TargetBlock
from landmark_detection import LandmarkDetection
from transformers import BertConfig, BertModel


class FashionVLP(nn.Module):
    """
    Mô hình FashionVLP hoàn chỉnh, bao gồm cả hai nhánh Reference và Target.
    """

    def __init__(
        self,
        d_model=768,
        resnet_model_name="resnet50",
        vlp_model_name="bert-base-uncased",
    ):
        super().__init__()
        # Feature Extractor (ResNet) - được chia sẻ giữa hai khối
        # Sử dụng timm để dễ dàng lấy feature không có lớp classifer cuối cùng
        # pretrained=True để fine-tune
        feature_extractor = timm.create_model(
            resnet_model_name, pretrained=True, num_classes=0, global_pool=""
        )
        self.d_img = feature_extractor.num_features  # Lấy số chiều feature từ ResNet

        vlp_config = BertConfig.from_pretrained(vlp_model_name, hidden_size=d_model)
        vlp_transformer = BertModel.from_pretrained(vlp_model_name, config=vlp_config)
        landmark_detection = LandmarkDetection()

        self.target_block = TargetBlock(
            feature_extractor, landmark_detection, self.d_img, d_model
        )
        self.reference_block = ReferenceBlock(
            feature_extractor, vlp_transformer, landmark_detection, self.d_img, d_model
        )

    def forward(self, ref_image, text_tokens, target_image):

        cropped_ref_image = self._get_crop(ref_image)
        cropped_target_image = self._get_crop(target_image)

        f_ref = self.reference_block(
            ref_image,
            cropped_ref_image,
            text_tokens
        )

        f_tar = self.target_block(target_image, cropped_target_image)
        # f_ref: [B, d_model], f_tar: [B, d_model]
        return f_ref, f_tar
