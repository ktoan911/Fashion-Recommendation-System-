import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LandmarkDetection:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, pil_image, min_detection_confidence=0.5, max_len=14):
        # Mở ảnh
        image_rgb = np.array(pil_image.convert("RGB"))

        with self.mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=min_detection_confidence
        ) as pose:
            results = pose.process(image_rgb)
            landmarks = (
                [
                    [lm.x, lm.y]
                    for lm in results.pose_landmarks.landmark
                    if 0 <= lm.x <= 1 and 0 <= lm.y <= 1
                ]
                if results.pose_landmarks
                else []
            )
        padded = [[0, 0] for _ in range(max_len)]
        for i, point in enumerate(landmarks):
            if i >= max_len:
                break
            padded[i] = point
        return padded

    def print_on_img(self, results, image_path):
        image = cv2.imread(image_path)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        cv2.imwrite("mediapipe_result.jpg", image)

    def get_landmark_features(
        self, feature_map: torch.Tensor, normalized_landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feature_map (torch.Tensor)
                Shape: [Batch, Channels, Height_fm, Width_fm]
            normalized_landmarks
                Shape: [Batch, Num_Landmarks, 2] với tọa độ (x, y).

        Returns:
            torch.Tensor:
                Shape: [Batch, Num_Landmarks, Channels]
        """
        batch_size, num_landmarks, _ = normalized_landmarks.shape

        # [0, 1] to [-1, 1] for grid_sample
        grid = normalized_landmarks * 2 - 1

        # F.grid_sample request[B, H_out, W_ot, 2].
        # [Batch, 1, Num_Landmarks, 2].
        grid = grid.view(batch_size, 1, num_landmarks, 2).to(device)

        sampled_features = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        # [Batch, Channels, 1, Num_Landmarks]

        # [B, C, 1, L] -> [B, C, L] -> [B, L, C]
        final_features = sampled_features.squeeze(2).permute(0, 2, 1)

        return final_features
