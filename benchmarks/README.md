# Torch-Molecule Benchmarks

This directory contains comprehensive benchmarking tools for torch-molecule models.

## Available Benchmarks

### 1. Quick Benchmark (`quick_benchmark.py`)
- **Purpose**: Fast proof-of-concept testing
- **Dataset**: Small subset (10-100 molecules)
- **Duration**: ~5-10 minutes
- **Use Case**: Testing model implementations and debugging

### 2. Full QM9 Benchmark (`full_qm9_benchmark.py`)
- **Purpose**: Complete evaluation on full dataset
- **Dataset**: Full QM9 (~133k molecules)
- **Duration**: 3-5 hours
- **Use Case**: Comprehensive model evaluation

### 3. **Cached Full QM9 Benchmark (`full_qm9_benchmark_cached.py`)** ⭐ **NEW**
- **Purpose**: Resumable full dataset benchmark with checkpointing
- **Dataset**: Full QM9 (~133k molecules) 
- **Duration**: 3-5 hours (but resumable if interrupted)
- **Use Case**: Production benchmarking with interruption safety

## 🆕 Cache System Features

### Automatic Checkpointing
- **Model States**: Saves trained model weights and parameters
- **Training Progress**: Saves loss curves and training metrics
- **Data Splits**: Caches train/val/test splits for consistency
- **Benchmark Progress**: Saves overall benchmark state

### Resume Functionality
```bash
# Resume from last checkpoint (default)
python benchmarks/full_qm9_benchmark_cached.py

# Start fresh but may reuse data splits
python benchmarks/full_qm9_benchmark_cached.py --no-resume

# Complete restart, ignore all cache
python benchmarks/full_qm9_benchmark_cached.py --force-restart
```

### Cache Management
```bash
# Check cache status
python benchmarks/cache_status.py

# Monitor running benchmark
python benchmarks/monitor_benchmark.py
```

## Dataset Splits

All benchmarks use consistent train/validation/test splits:
- **Training**: 80% (~107k molecules)
- **Validation**: 10% (~13k molecules)  
- **Testing**: 10% (~13k molecules)

## Models Tested

### 1. GNN Predictor
- **Architecture**: Graph Neural Network with virtual nodes
- **Task**: Molecular property prediction (QM9 gap)
- **Metrics**: MAE, R², RMSE

### 2. Supervised Encoder
- **Architecture**: Graph-based molecular encoder
- **Task**: Molecular representation learning
- **Metrics**: Embedding dimensionality and statistics

### 3. LSTM Generator
- **Architecture**: LSTM-based molecular generator
- **Task**: SMILES generation
- **Metrics**: Validity rate, sample molecules

## Cache Directory Structure

```
benchmarks/cache/
├── checkpoint_manager.py    # Cache management system
├── models/                  # Model checkpoints
│   ├── gnn_predictor_*.pkl
│   ├── supervised_encoder_*.pkl
│   └── lstm_generator_*.pkl
├── progress/               # Benchmark progress files
│   └── benchmark_progress_*.json
└── data/                   # Cached data splits
    └── data_split_*.pkl
```

## Monitoring and Status

### Quick Status Check
```bash
python benchmarks/quick_status.py
```

### Continuous Monitoring
```bash
python benchmarks/monitor_benchmark.py
```

### Process Management
```bash
# Check if benchmark is running
ps aux | grep benchmark

# Kill running benchmark (if needed)
pkill -f "full_qm9_benchmark"
```

## Results

All benchmarks save detailed JSON results with:
- **Model Performance**: MAE, R², RMSE, validity rates
- **Training Metrics**: Loss curves, training times, epochs
- **Dataset Statistics**: Sample molecules, target distributions
- **System Information**: Cache usage, duration, resource utilization

## Best Practices

### For Development/Testing
```bash
python benchmarks/quick_benchmark.py
```

### For Research/Production
```bash
# Start new benchmark
python benchmarks/full_qm9_benchmark_cached.py --force-restart

# Resume interrupted benchmark
python benchmarks/full_qm9_benchmark_cached.py
```

### For Continuous Integration
```bash
# Quick validation
python benchmarks/test_full_benchmark.py
```

## Cache Benefits

1. **Interruption Safety**: Resume from any point if interrupted
2. **Partial Results**: Access intermediate results even if not complete
3. **Reproducibility**: Consistent data splits across runs
4. **Time Savings**: Skip already-trained models
5. **Storage Efficiency**: Automatic cleanup of old checkpoints

## Example Usage

```bash
# 1. Test setup
python benchmarks/test_full_benchmark.py

# 2. Start cached benchmark
python benchmarks/full_qm9_benchmark_cached.py

# 3. Monitor progress (in another terminal)
python benchmarks/monitor_benchmark.py

# 4. Check cache status
python benchmarks/cache_status.py

# 5. Resume if interrupted
python benchmarks/full_qm9_benchmark_cached.py
```

## Troubleshooting

### Cache Issues
```bash
# Clear all cache
rm -rf benchmarks/cache/models/*
rm -rf benchmarks/cache/progress/*
rm -rf benchmarks/cache/data/*

# Force restart
python benchmarks/full_qm9_benchmark_cached.py --force-restart
```

### Memory Issues
- Reduce batch sizes in model configurations
- Use fewer training molecules for LSTM generator
- Monitor with `htop` or similar tools

### Import Errors
- Ensure torch-molecule is properly installed
- Check Python path and virtual environment
- Verify all dependencies are available