import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F


class LandmarkDetection:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, image_path, min_detection_confidence=0.5):
        # Mở ảnh
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_landmark_ids = set(range(0, 11))

        with self.mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=min_detection_confidence
        ) as pose:
            results = pose.process(image_rgb)
            return [
                [lm.x, lm.y]
                for idx, lm in enumerate(results.pose_landmarks.landmark)
                if idx not in face_landmark_ids and 0 <= lm.x <= 1 and 0 <= lm.y <= 1
            ]

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
        grid = grid.view(batch_size, 1, num_landmarks, 2)

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
