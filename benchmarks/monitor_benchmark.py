#!/usr/bin/env python3
"""
Monitor the progress of the full QM9 benchmark.
"""

import os
import time
import glob
from pathlib import Path

def monitor_benchmark():
    """Monitor the benchmark progress by checking output files."""
    print("📊 Monitoring full QM9 benchmark progress...")
    print("=" * 50)
    
    benchmark_dir = Path(__file__).parent
    
    while True:
        # Check for result files
        result_files = list(benchmark_dir.glob("full_qm9_benchmark_results_*.json"))
        
        if result_files:
            # Sort by modification time to get the latest
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            file_age = time.time() - latest_file.stat().st_mtime
            
            print(f"📄 Latest result file: {latest_file.name}")
            print(f"🕐 File age: {file_age/60:.1f} minutes")
            
            if file_age < 300:  # File updated within last 5 minutes
                print("✅ Benchmark appears to be running")
            else:
                print("⚠️  Benchmark may have finished or stalled")
                
                # Try to read the results
                try:
                    import json
                    with open(latest_file, 'r') as f:
                        results = json.load(f)
                    
                    print("\n📊 Current Results Summary:")
                    if "benchmark_info" in results:
                        duration = results["benchmark_info"].get("duration_seconds")
                        if duration:
                            print(f"  Duration: {duration/3600:.2f} hours")
                        
                        if "train_size" in results["benchmark_info"]:
                            print(f"  Train size: {results['benchmark_info']['train_size']:,}")
                    
                    if "results" in results:
                        for model, result in results["results"].items():
                            status = result.get("status", "unknown")
                            print(f"  {model}: {status}")
                            if status == "completed" and "training_time_minutes" in result:
                                print(f"    Training time: {result['training_time_minutes']:.1f} min")
                    
                except Exception as e:
                    print(f"  Error reading results: {e}")
                
                break
        else:
            print("📝 No result files found yet - benchmark may still be initializing...")
        
        print(f"\n⏰ {time.strftime('%H:%M:%S')} - Checking again in 30 seconds...")
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor_benchmark()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

