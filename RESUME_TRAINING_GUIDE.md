# Hướng dẫn Resume Training

## Tổng quan

Hệ thống Fashion Recommendation System đã được trang bị chức năng resume training, cho phép bạn tiếp tục training từ một checkpoint đã lưu trước đó.

## Cấu trúc Checkpoint

Mỗi checkpoint được lưu với cấu trúc dict như sau:

```python
{
    "epoch": epoch + 1,                    # Epoch hiện tại
    "model_state_dict": model.state_dict(), # Trạng thái model
    "optimizer_state_dict": optimizer.state_dict(), # Trạng thái optimizer
    "scheduler_state_dict": scheduler.state_dict(), # Trạng thái scheduler
    "loss": avg_loss,                      # Loss trung bình
    "recall_at_10": val_recall_10,         # Recall@10 trên validation
    "train_losses": train_losses,          # Lịch sử training losses
    "val_recalls": val_recalls,            # Lịch sử validation recalls
}
```

## Files đã tạo

1. **`resume_training.py`** - Script chính để resume training
2. **`run_resume_training.sh`** - Shell script để chạy resume training dễ dàng
3. **`test_resume.py`** - Script test chức năng resume training

## Cách sử dụng

### Phương pháp 1: Sử dụng Python script

```bash
# Kích hoạt virtual environment
source venv/bin/activate

# Resume training từ checkpoint
python resume_training.py \
    --checkpoint checkpoints/optimized_model_epoch_10_recall_0.3456.pt \
    --path-file /path/to/annotations \
    --path-folder /path/to/images \
    --batch 32 \
    --epochs 100 \
    --mixed-precision \
    --fast-validation
```

### Phương pháp 2: Sử dụng Shell script

```bash
# Cách đơn giản nhất
./run_resume_training.sh <checkpoint_path> <data_file> <data_folder>

# Ví dụ cụ thể
./run_resume_training.sh \
    checkpoints/optimized_model_epoch_10_recall_0.3456.pt \
    /path/to/annotations \
    /path/to/images \
    32 \
    100
```

## Tham số

### Tham số bắt buộc:
- `--checkpoint`: Đường dẫn đến checkpoint file
- `--path-file`: Đường dẫn đến thư mục annotations
- `--path-folder`: Đường dẫn đến thư mục images

### Tham số tùy chọn:
- `--batch`: Batch size (default: 64)
- `--epochs`: Tổng số epochs (default: 100)
- `--gradient-accumulation`: Gradient accumulation steps (default: 2)
- `--mixed-precision`: Bật mixed precision training
- `--fast-validation`: Sử dụng subset validation để đánh giá nhanh
- `--cache-dir`: Thư mục cache (default: "cache")
- `--lr`: Override learning rate (optional)

## Tính năng

### ✅ Đã implement:
1. **Load checkpoint hoàn chỉnh**: Model, optimizer, scheduler state
2. **Khôi phục training history**: Train losses và validation recalls
3. **Tiếp tục từ epoch chính xác**: Resume từ epoch đã dừng
4. **Best model tracking**: Tiếp tục tracking best model
5. **Mixed precision support**: Tương thích với mixed precision training
6. **Scheduler state recovery**: Khôi phục learning rate scheduler
7. **Gradient accumulation**: Hỗ trợ gradient accumulation
8. **Fast validation**: Validation nhanh cho debugging

### 🔧 Tính năng nâng cao:
1. **Automatic scheduler adjustment**: Tự động điều chỉnh scheduler nếu không có trong checkpoint
2. **Flexible learning rate**: Có thể override learning rate khi resume
3. **Cache support**: Sử dụng cached dataset để tăng tốc
4. **Error handling**: Xử lý lỗi khi checkpoint không tồn tại
5. **Progress tracking**: Hiển thị tiến trình training chi tiết

## Kiểm tra chức năng

Chạy test để đảm bảo chức năng hoạt động:

```bash
source venv/bin/activate
python test_resume.py
```

## Ví dụ thực tế

### Scenario 1: Resume từ epoch 50
```bash
# Training bị gián đoạn ở epoch 50
./run_resume_training.sh \
    checkpoints/optimized_model_epoch_50_recall_0.4123.pt \
    data/annotations \
    data/images \
    32 \
    100
```

### Scenario 2: Resume với settings khác
```bash
# Resume với batch size nhỏ hơn và learning rate thấp hơn
python resume_training.py \
    --checkpoint checkpoints/best_optimized_model_recall_0.4567.pt \
    --path-file data/annotations \
    --path-folder data/images \
    --batch 16 \
    --epochs 150 \
    --lr 1e-4 \
    --mixed-precision \
    --fast-validation
```

## Lưu ý quan trọng

1. **Checkpoint compatibility**: Đảm bảo checkpoint được tạo từ cùng model architecture
2. **Data paths**: Kiểm tra đường dẫn data chính xác
3. **GPU memory**: Điều chỉnh batch size phù hợp với GPU memory
4. **Virtual environment**: Luôn kích hoạt venv trước khi chạy
5. **Backup checkpoints**: Nên backup checkpoint quan trọng

## Troubleshooting

### Lỗi "Checkpoint not found"
```bash
# Kiểm tra file tồn tại
ls -la checkpoints/
```

### Lỗi "CUDA out of memory"
```bash
# Giảm batch size
python resume_training.py --checkpoint ... --batch 16
```

### Lỗi "Module not found"
```bash
# Kích hoạt virtual environment
source venv/bin/activate
```

## Kết quả

Sau khi resume training thành công, bạn sẽ có:
1. **Continued checkpoints**: `checkpoints/resumed_model_epoch_X_recall_Y.pt`
2. **Best model updates**: `checkpoints/best_resumed_model_recall_X.pt`
3. **Final model**: `checkpoints/final_resumed_model_epoch_X_recall_Y.pt`
4. **Training logs**: Chi tiết progress và metrics

---

**Chúc bạn training thành công! 🚀**