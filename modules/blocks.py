import torch
import torch.nn as nn

from modules.attention import (
    ContextualAttention,
    LandmarkAttention,
    PositionalAttention,
)


class TargetBlock(nn.Module):
    """
    Xử lý ảnh mục tiêu (target image) để tạo ra embedding ftar.
    Sử dụng các module attention để kết hợp các đặc trưng một cách hiệu quả.
    """

    def __init__(self, feature_extractor, landmark, d_img, d_model):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.d_img = d_img

        self.landmark_detection = landmark
        # Attention modules
        self.pos_attn_whole = PositionalAttention(d_img)
        self.pos_attn_crop = PositionalAttention(d_img)
        self.lmk_attn = LandmarkAttention(d_img)
        self.ctx_attn = ContextualAttention(d_img, d_img)

        # Lớp chiếu (projection) cuối cùng để khớp với không gian của f_ref
        self.projection = nn.Linear(d_img, d_model)

    def forward(self, target_image, cropped_image, landmark_locations):
        f_whole_spat = self.feature_extractor.forward_features(
            target_image
        )  # [B, C, H, W]
        # f_crop_spat: [B, C, H, W]
        f_crop_spat = self.feature_extractor.forward_features(cropped_image)

        f_lmk_spat = self._get_landmark_features(
            landmark_locations, f_whole_spat
        )  # [B, L, C]

        f_whole_glob = self.pos_attn_whole(f_whole_spat)  # [B, C]
        f_crop_glob = self.pos_attn_crop(f_crop_spat)  # [B, C]
        f_lmk_glob = self.lmk_attn(f_lmk_spat)  # [B, C]

        fused_target_rep = self.ctx_attn(
            f_whole_glob, f_crop_glob, f_lmk_glob
        )  # [B, C]

        f_tar = self.projection(fused_target_rep)  # [B, d_model]
        return f_tar

    def _get_landmark_features(self, landmark_locations, feature_map):
        return self.landmark_detection.get_landmark_features(
            feature_map, landmark_locations
        )


class ReferenceBlock(nn.Module):
    def __init__(self, feature_extractor, vlp_transformer, landmark, d_img, d_model):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.vlp_transformer = vlp_transformer
        self.landmark_detection = landmark
        # Các lớp chiếu để đưa các đặc trưng hình ảnh về chiều d_model
        self.img_feat_proj = nn.Linear(d_img, d_model)

    def forward(self, ref_image, cropped_image, text_tokens, landmark_locations):
        # f_whole_spat, f_crop_spat, f_lmk_spat
        f_whole_spat = self.feature_extractor.forward_features(
            ref_image
        )  # [B, C, H, W]
        f_crop_spat = self.feature_extractor.forward_features(
            cropped_image
        )  # [B, C, H, W]
        f_lmk_spat = self._get_landmark_features(
            landmark_locations, f_whole_spat
        )  # [B, L, C]

        #  Các chuỗi đặc trưng để đưa vào Transformer
        # Flatten các đặc trưng không gian và chiếu
        f_whole_seq = self.img_feat_proj(
            f_whole_spat.flatten(2).transpose(1, 2)
        )  # [B, H*W, d_model]
        f_crop_seq = self.img_feat_proj(
            f_crop_spat.flatten(2).transpose(1, 2)
        )  # [B, H*W, d_model]
        f_lmk_seq = self.img_feat_proj(f_lmk_spat)  # [B, L, d_model]

        # TODO: positional encoding RoI, whole, và cropped

        text_embeddings = self.vlp_transformer.embeddings(
            input_ids=text_tokens
        )  # [B, T, d_model] (T: max_text_len)

        # [CLS] text [SEP] img_feats
        combined_embeddings = torch.cat(
            [
                text_embeddings,
                f_whole_seq,
                f_crop_seq,
                f_lmk_seq,
            ],
            dim=1,
        )

        # Tạo attention mask tương ứng
        # (Cần cẩn thận khi tạo mask cho các phần khác nhau của chuỗi)
        attention_mask = torch.ones(
            combined_embeddings.shape[:2], device=combined_embeddings.device
        )

        transformer_outputs = self.vlp_transformer(
            inputs_embeds=combined_embeddings, attention_mask=attention_mask
        )

        # Lấy hidden state của token [CLS] làm embedding cuối cùng
        # f_ref là output của [CLS] token
        f_ref = transformer_outputs.last_hidden_state[:, 0, :]  # [B, d_model]
        return f_ref

    def _get_landmark_features(self, landmark_locations, feature_map):
        # Return [Batch, Num_Landmarks, Feature_Dim]
        return self.landmark_detection.get_landmark_features(
            feature_map, landmark_locations
        )
