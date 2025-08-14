#!/usr/bin/env python3
"""
Quick status check for the full benchmark.
"""

import os
import glob
import json
from pathlib import Path

def quick_status():
    """Check benchmark status quickly."""
    print("🔍 Quick Benchmark Status Check")
    print("=" * 40)
    
    # Check if process is running
    try:
        import subprocess
        result = subprocess.run(['pgrep', '-f', 'full_qm9_benchmark.py'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Benchmark process is running")
            print(f"   Process ID: {result.stdout.strip()}")
        else:
            print("❌ No benchmark process found")
    except:
        print("⚠️  Could not check process status")
    
    # Check for result files
    benchmark_dir = Path(__file__).parent
    result_files = list(benchmark_dir.glob("full_qm9_benchmark_results_*.json"))
    
    if result_files:
        latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
        print(f"📄 Latest result file: {latest_file.name}")
        
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            if "benchmark_info" in results:
                if "duration_seconds" in results["benchmark_info"]:
                    duration = results["benchmark_info"]["duration_seconds"]
                    print(f"⏱️  Duration: {duration/60:.1f} minutes")
                
                if "train_size" in results["benchmark_info"]:
                    print(f"📊 Training on: {results['benchmark_info']['train_size']:,} molecules")
            
            if "results" in results:
                print(f"📈 Model Progress:")
                for model, result in results["results"].items():
                    status = result.get("status", "unknown")
                    if status == "completed":
                        time_min = result.get("training_time_minutes", "unknown")
                        print(f"   ✅ {model}: {status} ({time_min} min)")
                    elif status == "failed":
                        print(f"   ❌ {model}: {status}")
                    else:
                        print(f"   🔄 {model}: in progress")
            
        except Exception as e:
            print(f"❌ Error reading results: {e}")
    else:
        print("📝 No result files found yet")
    
    print("\n💡 To monitor continuously: python benchmarks/monitor_benchmark.py")

if __name__ == "__main__":
    quick_status()

