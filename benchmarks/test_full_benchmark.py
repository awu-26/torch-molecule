#!/usr/bin/env python3
"""
Quick test of the full benchmark setup before running the complete version.
"""

import sys
from pathlib import Path

# Add torch-molecule to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_benchmark_setup():
    """Test that all imports and dataset loading works correctly."""
    print("🧪 Testing full benchmark setup...")
    
    # Test imports
    print("📦 Testing imports...")
    try:
        import torch_molecule
        print(f"✅ torch-molecule version: {getattr(torch_molecule, '__version__', 'unknown')}")
        
        from torch_molecule.predictor import gnn
        from torch_molecule.encoder import supervised
        from torch_molecule.generator import lstm
        from torch_molecule.datasets import load_qm9
        print("✅ All model imports successful")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test dataset loading
    print("📊 Testing dataset loading...")
    try:
        smiles_list, targets_array = load_qm9()
        print(f"✅ QM9 dataset loaded: {len(smiles_list)} molecules")
        print(f"  Target shape: {targets_array.shape}")
        print(f"  Sample SMILES: {smiles_list[:3]}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test train/test split
    print("🔄 Testing train/test split...")
    try:
        from sklearn.model_selection import train_test_split
        
        # Handle multi-target case
        if len(targets_array.shape) > 1 and targets_array.shape[1] == 1:
            targets_array = targets_array.flatten()
        
        # Create splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            smiles_list, targets_array, test_size=0.1, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1111, random_state=42  # 0.1/(1-0.1) ≈ 0.1111
        )
        
        print(f"✅ Train/Val/Test split successful:")
        print(f"  Train: {len(X_train):,} molecules")
        print(f"  Validation: {len(X_val):,} molecules") 
        print(f"  Test: {len(X_test):,} molecules")
        
    except Exception as e:
        print(f"❌ Train/test split failed: {e}")
        return False
    
    # Test model initialization (without training)
    print("🏗️ Testing model initialization...")
    try:
        # GNN Predictor
        gnn_model = gnn.GNNMolecularPredictor(
            num_task=1,
            num_layer=3,  # Smaller for test
            hidden_size=64,
            batch_size=32,
            epochs=1,  # Just 1 epoch for test
            verbose=False
        )
        print("✅ GNN model initialized")
        
        # Supervised Encoder
        encoder = supervised.SupervisedMolecularEncoder(
            hidden_size=64,
            num_layer=2,
            epochs=1,
            verbose=False
        )
        print("✅ Encoder initialized")
        
        # LSTM Generator
        generator = lstm.LSTMMolecularGenerator(
            hidden_size=128,
            num_layer=2,
            epochs=1,
            verbose=False
        )
        print("✅ Generator initialized")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Full benchmark should work correctly.")
    print("\n⚠️  Note: The full benchmark will take several hours to complete.")
    print("Consider running it overnight or on a machine with good computational resources.")
    
    return True

if __name__ == "__main__":
    test_benchmark_setup()

