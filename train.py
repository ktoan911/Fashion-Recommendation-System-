import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.fashioniq_dataset import FashionDataset

# Giả sử bạn đã định nghĩa mô hình FashionVLP theo kiến trúc trong bài báo
# Gồm 2 nhánh: Reference block và Target block.
# class FashionVLP(nn.Module): ...
from modules.fashion_vlp import (
    FashionVLP,  # Tưởng tượng có file model.py định nghĩa lớp này
)

parser = ArgumentParser()
parser.add_argument("--batch", default=64, type=int)
parser.add_argument("--path-file", required=True, type=str)
parser.add_argument("--path-folder", required=True, type=str)
args = parser.parse_args()


# Hàm mất mát: Batch-based classification loss, về bản chất là CrossEntropyLoss
# trên ma trận tương đồng cosine[cite: 211, 214].
# Nội năng (inner product) được dùng làm hàm kernel, tương đương cosine similarity
# khi các vector được chuẩn hóa[cite: 216].
def batch_classification_loss(f_ref, f_tar, temperature=0.07):
    """
    Triển khai hàm loss từ phương trình (6) trong bài báo.
    Đây là loss tương phản (contrastive loss) InfoNCE.
    """
    # f_ref: [B, D], f_tar: [B, D]  (B: batch size, D: số chiều embedding)
    B, D = f_ref.shape

    # Chuẩn hóa L2 các embedding để tính cosine similarity bằng tích vô hướng
    f_ref = F.normalize(f_ref, p=2, dim=1)
    f_tar = F.normalize(f_tar, p=2, dim=1)

    # Tính ma trận tương đồng (similarity matrix)
    # Tích vô hướng giữa mỗi f_ref và TẤT CẢ các f_tar trong batch
    # sim_matrix[i, j] = cos_sim(f_ref_i, f_tar_j)
    sim_matrix = torch.matmul(f_ref, f_tar.t()) / temperature  # -> [B, B]

    # Loss: cross-entropy giữa sim_matrix và ma trận đơn vị
    # Các cặp dương (positive pairs) nằm trên đường chéo chính (i, i)
    labels = torch.arange(B, device=f_ref.device)
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def calculate_recall_at_k(f_ref, f_tar, k=10):
    """
    Tính Recall@K (R@K) cho việc đánh giá hiệu suất recommendation.

    Args:
        f_ref: tensor [B, D] - embeddings của reference images
        f_tar: tensor [B, D] - embeddings của target images
        k: int - số lượng top candidates để tính recall

    Returns:
        recall_at_k: float - giá trị R@K
    """
    # Chuẩn hóa L2 các embeddings
    f_ref = F.normalize(f_ref, p=2, dim=1)
    f_tar = F.normalize(f_tar, p=2, dim=1)

    # Tính ma trận tương đồng cosine
    similarity_matrix = torch.matmul(f_ref, f_tar.t())  # [B, B]

    batch_size = similarity_matrix.size(0)
    correct_predictions = 0

    for i in range(batch_size):
        # Lấy độ tương đồng của sample i với tất cả target images
        similarities = similarity_matrix[i]

        # Sắp xếp theo thứ tự giảm dần và lấy top-k indices
        _, top_k_indices = torch.topk(similarities, k, largest=True)

        # Kiểm tra xem ground truth (index i) có trong top-k không
        if i in top_k_indices:
            correct_predictions += 1

    recall_at_k = correct_predictions / batch_size
    return recall_at_k


def evaluate_model(model, val_loader, device, k=10):
    """
    Đánh giá mô hình trên validation set với metric R@K.

    Args:
        model: mô hình FashionVLP
        val_loader: DataLoader cho validation set
        device: thiết bị tính toán (CPU/GPU)
        k: int - số lượng top candidates

    Returns:
        avg_recall_at_k: float - R@K trung bình trên toàn bộ validation set
    """
    model.eval()
    total_recall = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Chuyển dữ liệu lên GPU
            ref_images = batch["reference_image"].to(device)
            target_images = batch["target_image"].to(device)
            feedback_tokens = batch["feedback_tokens"].to(device)
            crop_reference_images = batch["crop_reference_image"].to(device)
            crop_target_images = batch["crop_target_image"].to(device)
            landmark_locations = batch["landmarks"]

            # Forward pass
            f_ref, f_tar = model(
                ref_images,
                feedback_tokens,
                target_images,
                crop_reference_images,
                crop_target_images,
                landmark_locations,
            )

            # Tính R@K cho batch này
            batch_recall = calculate_recall_at_k(f_ref, f_tar, k)
            total_recall += batch_recall
            num_batches += 1

    avg_recall_at_k = total_recall / num_batches if num_batches > 0 else 0
    return avg_recall_at_k


def load_model_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint từ file .pt

    Args:
        checkpoint_path: đường dẫn đến file checkpoint
        model: instance của model cần load weights
        optimizer: optimizer (optional) để tiếp tục training

    Returns:
        dict chứa thông tin checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Đã load model từ checkpoint: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"  Recall@10: {checkpoint.get('recall_at_10', 'N/A')}")

    return checkpoint


image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = FashionDataset(
    annotations_folder=args.path_file,
    folder_img=args.path_folder,
    transform=image_transform,
    type="train",
)
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

val_dataset = FashionDataset(
    annotations_folder=args.path_file,
    folder_img=args.path_folder,
    transform=image_transform,
    type="val",
)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionVLP().to(device)

optimizer = optim.Adam(model.parameters(), lr=4e-4, betas=(0.55, 0.999))


num_epochs = 100
validation_frequency = 5

# Thêm biến để track best model
best_recall = 0.0
best_model_path = None

train_losses = []
val_recalls = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        ref_images = batch["reference_image"].to(device)
        target_images = batch["target_image"].to(device)
        feedback_tokens = batch["feedback_tokens"].to(device)
        crop_reference_images = batch["crop_reference_image"].to(device)
        crop_target_images = batch["crop_target_image"].to(device)
        landmark_locations = batch["landmarks"]

        optimizer.zero_grad()

        f_ref, f_tar = model(
            ref_images,
            feedback_tokens,
            target_images,
            crop_reference_images,
            crop_target_images,
            landmark_locations,
        )

        loss = batch_classification_loss(f_ref, f_tar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    train_losses.append(avg_loss)

    if (epoch + 1) % validation_frequency == 0 or epoch == 0:
        print(f"\n--- Đánh giá Validation tại Epoch {epoch + 1} ---")

        val_recall_10 = evaluate_model(model, val_loader, device, k=10)
        val_recalls.append(val_recall_10)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Training Loss: {avg_loss:.4f}")
        print(f"  Validation R@10: {val_recall_10:.4f}")
        print("-" * 50)

        # Lưu model checkpoint định kỳ
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = (
            f"checkpoints/model_epoch_{epoch + 1}_recall_{val_recall_10:.4f}.pt"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "recall_at_10": val_recall_10,
                "train_losses": train_losses,
                "val_recalls": val_recalls,
            },
            checkpoint_path,
        )
        print(f"Model checkpoint đã được lưu: {checkpoint_path}")

        # Lưu best model nếu recall tốt hơn
        if val_recall_10 > best_recall:
            best_recall = val_recall_10
            best_model_path = f"checkpoints/best_model_recall_{best_recall:.4f}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "recall_at_10": val_recall_10,
                    "train_losses": train_losses,
                    "val_recalls": val_recalls,
                },
                best_model_path,
            )
            print(f"🎉 Best model được cập nhật: {best_model_path}")

    else:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

print("\n" + "=" * 60)
print("HOÀN THÀNH QUÁTRÌNH TRAINING")
print("=" * 60)
final_recall_10 = evaluate_model(model, val_loader, device, k=10)
print(f"Kết quả cuối cùng - Validation R@10: {final_recall_10:.4f}")
if val_recalls:
    best_recall_training = max(val_recalls)
    print(f"R@10 tốt nhất trong quá trình training: {best_recall_training:.4f}")

print(f"Training Loss cuối cùng: {train_losses[-1]:.4f}")

# Lưu model cuối cùng
os.makedirs("checkpoints", exist_ok=True)
final_model_path = (
    f"checkpoints/final_model_epoch_{num_epochs}_recall_{final_recall_10:.4f}.pt"
)
torch.save(
    {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss": train_losses[-1],
        "final_recall_at_10": final_recall_10,
        "train_losses": train_losses,
        "val_recalls": val_recalls,
        "best_recall": best_recall if val_recalls else final_recall_10,
    },
    final_model_path,
)
print(f"Final model đã được lưu: {final_model_path}")

# Lưu chỉ model weights (nhẹ hơn) để dễ load
model_weights_path = "checkpoints/fashion_vlp_weights.pt"
torch.save(model.state_dict(), model_weights_path)
print(f"Model weights đã được lưu: {model_weights_path}")

if best_model_path:
    print(f"Best model đã lưu tại: {best_model_path}")

print("=" * 60)
