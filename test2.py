import json
import os

# 1. Đường dẫn tới folder ảnh và file JSON
image_folder = "/media/DATA/Fashion-Recommendation-System-/data/images"
input_json = "/media/DATA/Fashion-Recommendation-System-/data/annotaions/val.json"
output_json = "val.json"

# 2. Lấy danh sách tên ảnh (bỏ phần đuôi)
image_names = {
    os.path.splitext(f)[0]
    for f in os.listdir(image_folder)
    if f.lower().endswith(".jpg")
}

# 3. Đọc dữ liệu JSON
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# 4. Lọc dữ liệu
filtered_data = [
    item
    for item in data
    if item["target"] in image_names and item["candidate"] in image_names
]

# 5. Lưu kết quả vào file mới
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Đã lọc xong, còn {len(filtered_data)} mẫu, lưu vào {output_json}")
