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

# Load environment variables
source .env

# Check MongoDB connection and data
echo "🔍 Checking MongoDB connection and data..."
python3 -c "
import pymongo
import os
import sys
from dotenv import load_dotenv

try:
    load_dotenv()
    
    mongo_uri = os.getenv('MONGO_URI', os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
    db_name = os.getenv('DB_NAME', 'fashion_db')
    collection_name = os.getenv('COLLECTION_NAME', 'features')
    
    print(f'DEBUG: mongo_uri = {mongo_uri}')
    print(f'DEBUG: db_name = {db_name}')
    print(f'DEBUG: collection_name = {collection_name}')

    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    
    # Check if features collection exists and has data
    collections = db.list_collection_names()
    if collection_name not in collections:
        print('MISSING_COLLECTION')
        sys.exit(1)
    
    feature_count = db[collection_name].count_documents({})
    if feature_count == 0:
        print('EMPTY_COLLECTION')
        sys.exit(1)
    
    print(f'SUCCESS: Found {feature_count} features in database')
    sys.exit(0) 
    
except Exception as e:
    print(f'CONNECTION_ERROR: {e}')
    sys.exit(1)
"

mongo_check_result=$?

if [ $mongo_check_result -ne 0 ]; then
    echo "⚠️  MongoDB check failed or no feature data found"
    echo "❓ Do you want to extract features from images and populate the database? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🔄 Extracting features from images..."
        
        # Check if extract script exists
        if [ ! -f "/media/DATA/Fashion-Recommendation-System-/datasets/extract_ftar_features.py" ]; then
            echo "❌ Feature extraction script not found: /media/DATA/Fashion-Recommendation-System-/datasets/extract_ftar_features.py"
            exit 1
        fi
        
        # Run feature extraction
        cd /media/DATA/Fashion-Recommendation-System-
        python3 datasets/extract_ftar_features.py --folder_path data/images --model_path model/fashion_vlp.pt --device cuda

        if [ $? -eq 0 ]; then
            echo "✅ Feature extraction completed successfully"
            cd - > /dev/null
        else
            echo "❌ Feature extraction failed"
            exit 1
        fi
    else
        echo "❌ Cannot proceed without feature data in MongoDB"
        echo "📝 Please run the feature extraction script manually or answer 'y' to extract automatically"
        exit 1
    fi
else
    echo "✅ MongoDB connection and data check passed"
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

echo "✅ All checks passed, starting application..."

# Run Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "🎉 Application has been started at: http://localhost:8501"