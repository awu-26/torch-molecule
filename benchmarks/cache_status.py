#!/usr/bin/env python3
"""
Cache status and management utility for torch-molecule benchmarks.
"""

import os
import glob
import json
from pathlib import Path

def main():
    """Show cache status and available commands."""
    print("💾 Torch-Molecule Benchmark Cache Manager")
    print("=" * 45)
    
    try:
        cache_dir = Path(__file__).parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Count different types of files
        checkpoints = list(cache_dir.glob("*_checkpoint.pkl"))
        results = list(cache_dir.glob("cached_qm9_benchmark_results_*.json"))
        
        # Calculate total size
        total_size = 0
        for file_path in cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"📁 Cache Directory: {cache_dir}")
        print(f"🏷️  Model Checkpoints: {len(checkpoints)}")
        print(f"📊 Result Files: {len(results)}")
        print(f"💿 Total Size: {total_size_mb:.1f} MB")
        
        # List available checkpoints
        if checkpoints:
            print(f"\n🔍 Available Checkpoints:")
            for checkpoint in checkpoints:
                model_name = checkpoint.stem.replace('_checkpoint', '')
                print(f"  {model_name}: {checkpoint.name}")
        else:
            print(f"\n📝 No checkpoints found")
        
        print(f"\n💡 Commands:")
        print(f"  python benchmarks/full_qm9_benchmark_cached.py --help")
        print(f"  python benchmarks/full_qm9_benchmark_cached.py          # Resume from cache")
        print(f"  python benchmarks/full_qm9_benchmark_cached.py --no-resume")
        print(f"  python benchmarks/full_qm9_benchmark_cached.py --force-restart")
        
    except Exception as e:
        print(f"❌ Error accessing cache: {e}")

if __name__ == "__main__":
    main()