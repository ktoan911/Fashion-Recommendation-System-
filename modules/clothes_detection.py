import cv2
import torch
from ultralytics import YOLO

feature_map = None


class ClothesDetection:
    def __init__(self, model_path=r"model/best.pt"):
        self.model = YOLO(model_path)
        self.model.eval()

    def hook(self, module, input, output):
        global feature_map
        feature_map = output[0] if isinstance(output, tuple) else output

    def extract_features(self, image_path):
        global feature_map
        feature_map = None

        # Yolov8 -> hook layer 8
        target_layer = self.model.model.model[8]
        handle = target_layer.register_forward_hook(self.hook)

        # Chạy prediction
        results = self.model(image_path, verbose=False)

        handle.remove()

        if feature_map is None:
            print("Error: Could not retrieve feature map from hook.")
            return []

        result = results[0]
        img_h, img_w = result.orig_shape

        _, fm_c, fm_h, fm_w = feature_map.shape

        detected_objects = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            object_tag = self.model.names[class_id]
            confidence = float(box.conf[0])

            bbox_coords = box.xyxy[0].cpu().numpy()

            x1 = int(bbox_coords[0] * (fm_w / img_w))
            y1 = int(bbox_coords[1] * (fm_h / img_h))
            x2 = int(bbox_coords[2] * (fm_w / img_w))
            y2 = int(bbox_coords[3] * (fm_h / img_h))

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fm_w, x2), min(fm_h, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            roi_feature_map = feature_map[:, :, y1:y2, x1:x2]

            f_roi = torch.mean(roi_feature_map, dim=[2, 3]).squeeze()

            detected_objects.append(
                {
                    "tag": object_tag,
                    "confidence": confidence,
                    "box": bbox_coords,
                    "feature_map": f_roi,
                }
            )

        feature_map = None
        return detected_objects

    def draw_bounding_boxes(
        self, image_path, detected_objects, output_path="output.jpg"
    ):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
            return

        for obj in detected_objects:
            box = obj["box"]
            x1, y1, x2, y2 = map(int, box)

            label = obj["tag"]
            confidence = obj["confidence"]

            text = f"{label}: {confidence:.2f}"

            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 2
            )  

            cv2.putText(
                image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        # Lưu ảnh kết quả
        cv2.imwrite(output_path, image)
        print(f"✅ Đã vẽ xong! Ảnh kết quả được lưu tại: {output_path}")


# if __name__ == "__main__":
#     # 1. Khởi tạo model
#     detector = ClothesDetection(
#         model_path="model/best.pt"
#     )  # Thay bằng đường dẫn model của bạn

#     # 2. Chọn ảnh để kiểm tra
#     image_to_test = "/media/DATA/FashionVLP/images.jpeg"  # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY

#     # 3. Trích xuất đặc trưng và phát hiện đối tượng
#     print(f"Đang xử lý ảnh: {image_to_test}...")
#     detections = detector.extract_features(image_to_test)
#     print(f"Tìm thấy {len(detections)} đối tượng.")

#     # 4. Vẽ bounding box lên ảnh
#     if detections:
#         draw_bounding_boxes(
#             image_path=image_to_test,
#             detected_objects=detections,
#             output_path="test_result.jpg",  # Tên file ảnh output
#         )
#     else:
#         print("Không tìm thấy đối tượng nào để vẽ.")
