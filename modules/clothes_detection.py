from PIL import Image
from ultralytics import YOLO


class ClothesDetection:
    def __init__(self, model_path="model/yolov8n_finetune.pt"):
        self.model = YOLO(model_path)

    def get_highest_confidence_object(self, pil_image):
        results = self.model(pil_image, conf=0.25, iou=0.7, verbose=False)

        best_confidence = -1
        best_bbox = None

        for r in results:
            boxes = r.boxes

            if len(boxes) > 0:
                for box in boxes:
                    bbox_xyxy = box.xyxy[0].tolist()
                    confidence = box.conf.item()

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_bbox = bbox_xyxy

        if best_bbox:
            cropped_image = pil_image.crop(best_bbox)
            return cropped_image
        else:
            return None

    def print_img(self, ref_img_path):
        original_image = Image.open(ref_img_path).convert("RGB")
        highest_conf_object_image = self.get_highest_confidence_object(original_image)
        if highest_conf_object_image:
            # Lưu ảnh kết quả (nếu tìm thấy)
            output_path = "highest_confidence_object.jpg"
            highest_conf_object_image.save(output_path)

        else:
            print("Error object detection")
