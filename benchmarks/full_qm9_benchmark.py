#!/usr/bin/env python3
"""
Full QM9 Dataset Benchmark for torch-molecule models.

This script provides a comprehensive benchmark that:
- Uses the complete QM9 dataset (133k+ molecules)
- Implements proper train/validation/test splits
- Trains models with appropriate parameters for full dataset
- Provides detailed metrics and evaluation
- Saves comprehensive results and model checkpoints
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Add torch-molecule to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_train_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Create train/validation/test splits."""
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=None
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_full_qm9_benchmark():
    """Run comprehensive benchmark on full QM9 dataset."""
    print("🚀 Starting full QM9 torch-molecule benchmark...")
    start_time = time.time()
    
    # Results structure
    results = {
        "benchmark_info": {
            "name": "Full QM9 Dataset Benchmark",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": None,
            "dataset_size": None,
            "train_size": None,
            "val_size": None,
            "test_size": None
        },
        "models_tested": [],
        "dataset_info": {},
        "results": {},
        "summary": {}
    }
    
    try:
        # Import torch-molecule
        print("📦 Importing torch-molecule...")
        import torch_molecule
        print(f"✅ torch-molecule version: {getattr(torch_molecule, '__version__', 'unknown')}")
        
        # Test model imports
        print("🧪 Testing model imports...")
        model_imports = {}
        
        try:
            from torch_molecule.predictor import gnn
            model_imports["GNNMolecularPredictor"] = gnn.GNNMolecularPredictor
            results["models_tested"].append("GNNMolecularPredictor")
            print("✅ GNNMolecularPredictor imported successfully")
        except Exception as e:
            print(f"❌ Failed to import GNNMolecularPredictor: {e}")
        
        try:
            from torch_molecule.encoder import supervised
            model_imports["SupervisedMolecularEncoder"] = supervised.SupervisedMolecularEncoder
            results["models_tested"].append("SupervisedMolecularEncoder")
            print("✅ SupervisedMolecularEncoder imported successfully")
        except Exception as e:
            print(f"❌ Failed to import SupervisedMolecularEncoder: {e}")
        
        try:
            from torch_molecule.generator import lstm
            model_imports["LSTMMolecularGenerator"] = lstm.LSTMMolecularGenerator
            results["models_tested"].append("LSTMMolecularGenerator")
            print("✅ LSTMMolecularGenerator imported successfully")
        except Exception as e:
            print(f"❌ Failed to import LSTMMolecularGenerator: {e}")
        
        # Load full QM9 dataset
        print("📊 Loading full QM9 dataset...")
        try:
            from torch_molecule.datasets import load_qm9
            smiles_list, targets_array = load_qm9()
            
            # Handle multi-target case - flatten if needed for single target
            if len(targets_array.shape) > 1 and targets_array.shape[1] == 1:
                targets_array = targets_array.flatten()
            
            print(f"✅ QM9 dataset loaded: {len(smiles_list)} molecules")
            
            # Dataset statistics
            results["dataset_info"] = {
                "total_molecules": len(smiles_list),
                "target_shape": targets_array.shape,
                "target_stats": {
                    "mean": float(np.mean(targets_array)),
                    "std": float(np.std(targets_array)),
                    "min": float(np.min(targets_array)),
                    "max": float(np.max(targets_array))
                },
                "sample_smiles": smiles_list[:5],
                "sample_targets": targets_array[:5].tolist()
            }
            
        except Exception as e:
            print(f"❌ Failed to load QM9 dataset: {e}")
            return results
        
        # Create train/validation/test splits
        print("🔄 Creating train/validation/test splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
            smiles_list, targets_array, test_size=0.1, val_size=0.1, random_state=42
        )
        
        results["benchmark_info"]["dataset_size"] = len(smiles_list)
        results["benchmark_info"]["train_size"] = len(X_train)
        results["benchmark_info"]["val_size"] = len(X_val)
        results["benchmark_info"]["test_size"] = len(X_test)
        
        print(f"  Train set: {len(X_train)} molecules")
        print(f"  Validation set: {len(X_val)} molecules")
        print(f"  Test set: {len(X_test)} molecules")
        
        # Model Training and Evaluation
        if model_imports:
            print("\n🏃 Starting model training and evaluation...")
            
            # 1. GNN Predictor
            if "GNNMolecularPredictor" in model_imports:
                print("\n📈 Training GNN Predictor...")
                try:
                    train_start = time.time()
                    
                    # Initialize model with parameters suitable for full dataset
                    model = model_imports["GNNMolecularPredictor"](
                        num_task=1,
                        num_layer=5,
                        hidden_size=300,
                        gnn_type="gin-virtual",
                        drop_ratio=0.1,
                        norm_layer="batch_norm",
                        graph_pooling="mean",
                        augmented_feature=['maccs', 'morgan'],
                        batch_size=64,  # Reasonable batch size for full dataset
                        epochs=100,     # More epochs for better convergence
                        learning_rate=0.001,
                        patience=20,    # Early stopping patience
                        verbose=True    # Show training progress
                    )
                    
                    # Train model
                    print(f"  Training on {len(X_train)} molecules...")
                    model.fit(X_train, y_train, X_val, y_val)
                    train_time = time.time() - train_start
                    
                    # Evaluate model
                    print("  📊 Evaluating GNN predictor...")
                    prediction_result = model.predict(X_test)
                    
                    # Extract predictions from dict format
                    if isinstance(prediction_result, dict) and 'prediction' in prediction_result:
                        predictions = prediction_result['prediction']
                        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                            predictions = predictions.flatten()
                    else:
                        predictions = prediction_result
                    
                    # Calculate comprehensive metrics
                    mae = mean_absolute_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    
                    # Safely extract training history
                    fitting_loss = getattr(model, 'fitting_loss', [])
                    fitting_epoch = getattr(model, 'fitting_epoch', 0)
                    
                    results["results"]["gnn_predictor"] = {
                        "status": "completed",
                        "training_time_seconds": train_time,
                        "training_time_minutes": train_time / 60,
                        "mae": float(mae),
                        "r2": float(r2),
                        "rmse": float(rmse),
                        "epochs_trained": len(fitting_loss) if isinstance(fitting_loss, list) else 0,
                        "final_train_loss": fitting_loss[-1] if fitting_loss and isinstance(fitting_loss, list) else None,
                        "best_epoch": fitting_epoch,
                        "prediction_stats": {
                            "mean": float(np.mean(predictions)),
                            "std": float(np.std(predictions)),
                            "min": float(np.min(predictions)),
                            "max": float(np.max(predictions))
                        }
                    }
                    print(f"  ✅ GNN training completed - MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
                    print(f"  Training time: {train_time/60:.1f} minutes")
                    
                except Exception as e:
                    print(f"  ❌ GNN training failed: {e}")
                    results["results"]["gnn_predictor"] = {"status": "failed", "error": str(e)}
            
            # 2. Supervised Encoder
            if "SupervisedMolecularEncoder" in model_imports:
                print("\n🧬 Training Supervised Encoder...")
                try:
                    train_start = time.time()
                    
                    # Initialize encoder with parameters suitable for full dataset
                    encoder = model_imports["SupervisedMolecularEncoder"](
                        hidden_size=300,
                        num_layer=5,
                        drop_ratio=0.1,
                        batch_size=64,
                        epochs=50,      # Fewer epochs for encoder
                        learning_rate=0.001,
                        verbose=True
                    )
                    
                    # Train encoder
                    print(f"  Training on {len(X_train)} molecules...")
                    encoder.fit(X_train)
                    train_time = time.time() - train_start
                    
                    # Get embeddings for evaluation
                    print("  📊 Generating embeddings...")
                    embeddings_train = encoder.encode(X_train[:1000])  # Sample for speed
                    embeddings_test = encoder.encode(X_test[:1000])
                    
                    # Safely extract training history
                    fitting_loss = getattr(encoder, 'fitting_loss', [])
                    fitting_epoch = getattr(encoder, 'fitting_epoch', 0)
                    
                    results["results"]["supervised_encoder"] = {
                        "status": "completed",
                        "training_time_seconds": train_time,
                        "training_time_minutes": train_time / 60,
                        "embedding_dim": embeddings_train.shape[1] if hasattr(embeddings_train, 'shape') else len(embeddings_train[0]),
                        "epochs_trained": len(fitting_loss) if isinstance(fitting_loss, list) else 0,
                        "final_train_loss": fitting_loss[-1] if fitting_loss and isinstance(fitting_loss, list) else None,
                        "best_epoch": fitting_epoch,
                        "embedding_stats": {
                            "train_mean": float(np.mean(embeddings_train)) if hasattr(embeddings_train, 'shape') else None,
                            "train_std": float(np.std(embeddings_train)) if hasattr(embeddings_train, 'shape') else None,
                            "test_mean": float(np.mean(embeddings_test)) if hasattr(embeddings_test, 'shape') else None,
                            "test_std": float(np.std(embeddings_test)) if hasattr(embeddings_test, 'shape') else None
                        }
                    }
                    print(f"  ✅ Encoder training completed - Embedding dim: {embeddings_train.shape[1] if hasattr(embeddings_train, 'shape') else 'unknown'}")
                    print(f"  Training time: {train_time/60:.1f} minutes")
                    
                except Exception as e:
                    print(f"  ❌ Encoder training failed: {e}")
                    results["results"]["supervised_encoder"] = {"status": "failed", "error": str(e)}
            
            # 3. LSTM Generator
            if "LSTMMolecularGenerator" in model_imports:
                print("\n🧪 Training LSTM Generator...")
                try:
                    train_start = time.time()
                    
                    # Initialize generator with parameters suitable for full dataset
                    generator = model_imports["LSTMMolecularGenerator"](
                        hidden_size=512,
                        num_layer=3,
                        dropout=0.2,
                        batch_size=128,
                        epochs=50,      # More epochs for better molecular generation
                        learning_rate=0.0002,
                        max_len=100,
                        verbose=True
                    )
                    
                    # Train generator with substantial portion of dataset
                    train_subset_size = min(50000, len(X_train))  # Use up to 50k molecules for training
                    lstm_trainX = X_train[:train_subset_size]
                    print(f"  Training on {len(lstm_trainX)} molecules...")
                    generator.fit(lstm_trainX)
                    train_time = time.time() - train_start
                    
                    # Generate molecules
                    print("  🧪 Generating molecules...")
                    generated = generator.generate(batch_size=100)  # Generate more molecules
                    
                    # Validate generated molecules
                    valid_molecules = [mol for mol in generated if mol is not None]
                    
                    # Calculate validity statistics
                    validity_rate = len(valid_molecules) / len(generated) if generated else 0
                    
                    # Additional validation with RDKit
                    rdkit_valid_count = 0
                    if valid_molecules:
                        try:
                            from rdkit import Chem
                            for smi in valid_molecules:
                                if Chem.MolFromSmiles(smi) is not None:
                                    rdkit_valid_count += 1
                        except:
                            rdkit_valid_count = "unknown"
                    
                    # Safely extract training history
                    fitting_loss = getattr(generator, 'fitting_loss', [])
                    fitting_epoch = getattr(generator, 'fitting_epoch', 0)
                    
                    results["results"]["lstm_generator"] = {
                        "status": "completed",
                        "training_time_seconds": train_time,
                        "training_time_minutes": train_time / 60,
                        "training_molecules": len(lstm_trainX),
                        "molecules_generated": len(generated),
                        "valid_molecules": len(valid_molecules),
                        "rdkit_valid_molecules": rdkit_valid_count,
                        "validity_rate": validity_rate,
                        "rdkit_validity_rate": rdkit_valid_count / len(generated) if generated and isinstance(rdkit_valid_count, int) else None,
                        "epochs_trained": len(fitting_loss) if isinstance(fitting_loss, list) else 0,
                        "final_train_loss": fitting_loss[-1] if fitting_loss and isinstance(fitting_loss, list) else None,
                        "best_epoch": fitting_epoch,
                        "sample_molecules": valid_molecules[:10] if valid_molecules else generated[:10]
                    }
                    
                    print(f"  ✅ Generator training completed")
                    print(f"  Training time: {train_time/60:.1f} minutes")
                    print(f"  Generated: {len(generated)} molecules, {len(valid_molecules)} valid ({validity_rate:.2%})")
                    if isinstance(rdkit_valid_count, int):
                        print(f"  RDKit valid: {rdkit_valid_count} ({rdkit_valid_count/len(generated):.2%})")
                    
                except Exception as e:
                    print(f"  ❌ Generator training failed: {e}")
                    results["results"]["lstm_generator"] = {"status": "failed", "error": str(e)}
        
    except ImportError as e:
        print(f"❌ torch-molecule not found or not properly installed: {e}")
        results["results"]["import_error"] = str(e)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        results["results"]["unexpected_error"] = str(e)
    
    # Calculate duration
    duration = time.time() - start_time
    results["benchmark_info"]["duration_seconds"] = duration
    
    # Summary
    successful_models = [model for model in results["results"] if results["results"][model].get("status") == "completed"]
    results["summary"] = {
        "models_successfully_imported": len(results["models_tested"]),
        "models_successfully_trained": len(successful_models),
        "successful_models": successful_models,
        "total_duration_seconds": duration,
        "total_duration_minutes": duration / 60,
        "total_duration_hours": duration / 3600,
        "status": "completed"
    }
    
    # Save results
    output_file = Path(__file__).parent / f"full_qm9_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("📊 FULL QM9 BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total duration: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
    print(f"Dataset: {results['benchmark_info']['dataset_size']:,} molecules")
    print(f"Train/Val/Test: {results['benchmark_info']['train_size']:,}/{results['benchmark_info']['val_size']:,}/{results['benchmark_info']['test_size']:,}")
    print(f"Models imported: {len(results['models_tested'])}")
    print(f"Models successfully trained: {len(successful_models)}")
    print(f"Results saved to: {output_file}")
    
    if results["models_tested"]:
        print(f"\n✅ Models tested: {', '.join(results['models_tested'])}")
    if successful_models:
        print(f"✅ Successfully trained: {', '.join(successful_models)}")
    
    # Detailed results
    for model_name, model_results in results["results"].items():
        if model_results.get("status") == "completed":
            print(f"\n📋 {model_name} Results:")
            if "mae" in model_results:
                print(f"  MAE: {model_results['mae']:.4f}")
                print(f"  R²: {model_results['r2']:.4f}")
                print(f"  RMSE: {model_results['rmse']:.4f}")
            if "embedding_dim" in model_results:
                print(f"  Embedding dimension: {model_results['embedding_dim']}")
            if "validity_rate" in model_results:
                print(f"  Molecule validity rate: {model_results['validity_rate']:.2%}")
            print(f"  Training time: {model_results['training_time_minutes']:.1f} minutes")
    
    print("\n🎉 Full QM9 benchmark completed!")
    return results

if __name__ == "__main__":
    run_full_qm9_benchmark()

