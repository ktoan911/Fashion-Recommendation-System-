#!/usr/bin/env python3
"""
Script kiểm tra GPU và đưa ra khuyến nghị cấu hình tối ưu cho GPU 15GB
"""

import torch
import subprocess
import sys

def get_gpu_info():
    """Lấy thông tin GPU"""
    if not torch.cuda.is_available():
        print("❌ CUDA không available!")
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return {
        'name': gpu_name,
        'memory_gb': gpu_memory_gb,
        'memory_mb': gpu_memory_gb * 1024
    }

def get_nvidia_smi_info():
    """Lấy thông tin từ nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return None
    except FileNotFoundError:
        return None

def recommend_config(gpu_memory_gb):
    """Đưa ra khuyến nghị cấu hình dựa trên GPU memory"""
    
    configs = []
    
    if gpu_memory_gb >= 24:
        configs.append({
            'name': '🚀 High-End GPU (24GB+)',
            'batch': 128,
            'grad_accum': 1,
            'effective_batch': 128,
            'memory_usage': '~18GB',
            'recommended': True
        })
    
    if gpu_memory_gb >= 15:
        configs.extend([
            {
                'name': '🎯 Optimal for 15GB',
                'batch': 64,
                'grad_accum': 2, 
                'effective_batch': 128,
                'memory_usage': '~14GB',
                'recommended': True
            },
            {
                'name': '🚀 Fast Training',
                'batch': 32,
                'grad_accum': 4,
                'effective_batch': 128, 
                'memory_usage': '~12GB',
                'recommended': True
            },
            {
                'name': '💾 Conservative',
                'batch': 16,
                'grad_accum': 8,
                'effective_batch': 128,
                'memory_usage': '~6GB',
                'recommended': False
            }
        ])
    
    if gpu_memory_gb >= 8:
        configs.append({
            'name': '⚠️ Memory Limited',
            'batch': 8,
            'grad_accum': 16,
            'effective_batch': 128,
            'memory_usage': '~3GB',
            'recommended': gpu_memory_gb < 12
        })
    
    return configs

def estimate_training_time(config, dataset_size=10000):
    """Ước tính thời gian training"""
    batches_per_epoch = dataset_size // config['effective_batch']
    
    # Estimates based on typical performance
    seconds_per_batch = {
        8: 2.5,   # Small batch slower per sample
        16: 2.0,
        32: 1.5,  # Optimal efficiency
        64: 1.8,  # Approaching memory limit
        128: 2.2  # Near OOM, slower
    }.get(config['batch'], 2.0)
    
    epoch_time_minutes = (batches_per_epoch * seconds_per_batch) / 60
    
    return {
        'batches_per_epoch': batches_per_epoch,
        'epoch_time_minutes': epoch_time_minutes,
        'total_time_hours_100_epochs': epoch_time_minutes * 100 / 60
    }

def main():
    print("🔍 FASHION VLP GPU CONFIGURATION CHECKER")
    print("=" * 50)
    
    # Get GPU info
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ Không thể detect GPU. Vui lòng kiểm tra CUDA installation.")
        return
    
    print(f"🎮 GPU: {gpu_info['name']}")
    print(f"💾 Memory: {gpu_info['memory_gb']:.1f} GB")
    print()
    
    # Get nvidia-smi info
    nvidia_info = get_nvidia_smi_info()
    if nvidia_info:
        print("📊 NVIDIA-SMI Output:")
        print("-" * 30)
        # Extract relevant lines
        lines = nvidia_info.split('\n')
        for line in lines[2:8]:  # GPU info section
            if line.strip():
                print(line)
        print()
    
    # Get recommendations
    configs = recommend_config(gpu_info['memory_gb'])
    
    print("🎯 KHUYẾN NGHỊ CONFIGURATION:")
    print("=" * 50)
    
    for i, config in enumerate(configs, 1):
        status = "⭐ RECOMMENDED" if config['recommended'] else "   Alternative"
        print(f"{i}. {config['name']} {status}")
        print(f"   --batch {config['batch']} --gradient-accumulation {config['grad_accum']}")
        print(f"   Effective batch: {config['effective_batch']}")
        print(f"   Estimated memory: {config['memory_usage']}")
        
        # Training time estimate
        timing = estimate_training_time(config)
        print(f"   Est. time/epoch: {timing['epoch_time_minutes']:.1f} min")
        print(f"   Est. 100 epochs: {timing['total_time_hours_100_epochs']:.1f} hours")
        print()
    
    # Command examples
    print("🚀 COMMAND EXAMPLES:")
    print("=" * 50)
    
    recommended_configs = [c for c in configs if c['recommended']]
    
    for config in recommended_configs[:2]:  # Top 2 recommendations
        print(f"# {config['name']}")
        print("python train_optimized.py \\")
        print("    --path-file /path/to/annotations \\")
        print("    --path-folder /path/to/images \\")
        print(f"    --batch {config['batch']} \\")
        print(f"    --gradient-accumulation {config['grad_accum']} \\")
        print("    --mixed-precision \\")
        print("    --fast-validation")
        print()
    
    # Memory monitoring tips
    print("💡 MEMORY MONITORING TIPS:")
    print("=" * 50)
    print("1. 📊 Monitor GPU usage: watch -n 1 nvidia-smi")
    print("2. 🔧 Start with smaller batch, increase gradually")
    print("3. ⚠️ If OOM, reduce batch size and increase gradient accumulation")
    print("4. 🚀 Mixed precision saves ~50% memory")
    print("5. 💾 Cache will use some system RAM but saves GPU memory")
    
    # Specific advice for 15GB
    if 14 <= gpu_info['memory_gb'] <= 16:
        print("\n🎯 SPECIFIC ADVICE FOR YOUR 15GB GPU:")
        print("=" * 50)
        print("✅ Perfect size for Fashion VLP training!")
        print("✅ Can use batch_size=64 with mixed precision")
        print("✅ Recommend: --batch 32 --gradient-accumulation 4")
        print("⚠️ Avoid batch_size > 64 to prevent OOM")
        print("💡 If training other models simultaneously, use --batch 16")

if __name__ == "__main__":
    main()