#!/bin/bash

# 🚀 Script chạy training optimized với các cấu hình khác nhau

echo "=== FASHION VLP OPTIMIZED TRAINING ==="
echo "Chọn cấu hình training:"
echo "1. 🚀 Maximum Speed (Mixed Precision + Fast Validation)"
echo "2. 💾 Memory Efficient (Small batch + Gradient Accumulation)" 
echo "3. 🎯 Best Accuracy (Large batch + Full validation)"
echo "4. 🧪 Benchmark Performance"
echo "5. 📊 Custom Configuration"

read -p "Nhập lựa chọn (1-5): " choice

# Kiểm tra arguments required
if [[ -z "$1" ]] || [[ -z "$2" ]]; then
    echo "❌ Thiếu arguments!"
    echo "Usage: $0 <path_to_annotations> <path_to_images>"
    echo "Example: $0 /path/to/annotations /path/to/images"
    exit 1
fi

PATH_FILE=$1
PATH_FOLDER=$2
BATCH=$3
EPOCHS=$4

echo "🚀 Running Maximum Speed Configuration..."
python train_optimized.py \
            --path-file "$PATH_FILE" \
            --path-folder "$PATH_FOLDER" \
            --batch $BATCH \
            --gradient-accumulation 4 \
            --mixed-precision \
            --fast-validation \
            --epochs $EPOCHS \
            --checkpoint-interval 5
        ;;


echo "✅ Training completed! Check checkpoints/ folder for saved models."
echo "📊 View OPTIMIZATION_GUIDE.md for more details."