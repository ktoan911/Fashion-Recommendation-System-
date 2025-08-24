#!/bin/bash

# Script to run the Streamlit Fashion Recommendation application

echo "🚀 Starting Fashion Recommendation System..."

# Check .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found"
    echo "📝 Please copy .env.example to .env and configure MongoDB information"
    echo "   cp .env.example .env"
    echo "   nano .env"
    exit 1
fi

# Check model files
if [ ! -f "model/fashion_vlp.pt" ]; then
    echo "⚠️  Model file not found: model/fashion_vlp.pt"
    echo "📝 Please ensure the model has been trained and placed in the correct location"
    exit 1
fi

if [ ! -f "model/yolov8n_finetune.pt" ]; then
    echo "⚠️  YOLO model not found: model/yolov8n_finetune.pt"
    echo "📝 Please ensure the YOLO model has been placed in the correct location"
    exit 1
fi

if [ ! -d "data/images" ]; then
    echo "⚠️  Images directory not found: data/images"
    echo "📝 Please ensure the images directory exists and contains data"
    exit 1
fi

echo "✅ Checks passed, starting application..."

# Run Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "🎉 Application has been started at: http://localhost:8501"
