#!/bin/bash

PATH_FILE=$1
PATH_FOLDER=$2
BATCH=$3
EPOCHS=$4

echo "🚀 Running Maximum Speed Configuration..."
python train.py \
            --path-file "$PATH_FILE" \
            --path-folder "$PATH_FOLDER" \
            --batch $BATCH \
            --gradient-accumulation 4 \
            --mixed-precision \
            --fast-validation \
            --epochs $EPOCHS \

echo "✅ Training completed!"