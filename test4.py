import os

folder_path = "/media/DATA/Fashion-Recommendation-System-/data/images"

# Các đuôi ảnh muốn tính
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

count = sum(
    1
    for file in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, file))
    and os.path.splitext(file)[1].lower() in image_exts
)

print(f"Số ảnh: {count}")
