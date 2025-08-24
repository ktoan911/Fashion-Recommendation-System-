import timm  # Thư viện hữu ích cho các backbone CNN
import torch.nn as nn
from transformers import BertConfig, BertModel

from modules.blocks import ReferenceBlock, TargetBlock
from modules.landmark_detection import LandmarkDetection


class FashionVLP(nn.Module):
    def __init__(
        self,
        d_model=768,
        resnet_model_name="resnet50",
        vlp_model_name="bert-base-uncased",
    ):
        super().__init__()
        feature_extractor = timm.create_model(
            resnet_model_name, pretrained=True, num_classes=0, global_pool=""
        )
        self.d_img = feature_extractor.num_features 
        vlp_config = BertConfig.from_pretrained(vlp_model_name, hidden_size=d_model)
        vlp_transformer = BertModel.from_pretrained(vlp_model_name, config=vlp_config)
        landmark_detection = LandmarkDetection()

        self.target_block = TargetBlock(
            feature_extractor, landmark_detection, self.d_img, d_model
        )
        self.reference_block = ReferenceBlock(
            feature_extractor, vlp_transformer, landmark_detection, self.d_img, d_model
        )

    def forward(
        self,
        ref_image,
        text_tokens,
        target_image,
        crop_reference_image,
        crop_target_image,
        landmark_locations,
    ):
        f_ref = self.reference_block(
            ref_image, crop_reference_image, text_tokens, landmark_locations
        )

        f_tar = self.target_block(target_image, crop_target_image, landmark_locations)
        # f_ref: [B, d_model], f_tar: [B, d_model]
        return f_ref, f_tar
