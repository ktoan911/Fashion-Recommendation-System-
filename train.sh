#!/bin/bash

# 🚀 Script chạy training optimized với các cấu hình khác nhau

echo "=== FASHION VLP OPTIMIZED TRAINING ==="

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