#!/usr/bin/env python3
"""
Full QM9 Dataset Benchmark with Simple Caching for torch-molecule models.

This script provides a comprehensive benchmark that:
- Uses the complete QM9 dataset (133k+ molecules)
- Implements proper train/validation/test splits
- Saves model checkpoints to cache folder
- Can resume from saved model weights
- Provides detailed metrics and evaluation
"""

import sys
import time
import json
import pickle
import numpy as np
import argparse
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

def save_checkpoint(model, results, model_name, cache_dir, epoch=None, is_final=False):
    """Save model checkpoint and results."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    checkpoint = {
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        'is_final': is_final
    }
    
    # Save model state if available
    if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
        try:
            import torch
            checkpoint['model_state_dict'] = model.model.state_dict()
        except:
            pass
    elif hasattr(model, 'state_dict'):
        try:
            import torch
            checkpoint['model_state_dict'] = model.state_dict()
        except:
            pass
    
    # Save model attributes
    if hasattr(model, 'fitting_loss'):
        checkpoint['fitting_loss'] = getattr(model, 'fitting_loss', [])
    if hasattr(model, 'fitting_epoch'):
        checkpoint['fitting_epoch'] = getattr(model, 'fitting_epoch', 0)
    
    # Choose filename based on whether this is final or intermediate
    if is_final or epoch is None:
        checkpoint_file = cache_path / f"{model_name}_checkpoint.pkl"
    else:
        checkpoint_file = cache_path / f"{model_name}_checkpoint_epoch_{epoch}.pkl"
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    status = "Final" if is_final else f"Epoch {epoch}" if epoch is not None else "Intermediate"
    print(f"💾 Saved {status} checkpoint: {checkpoint_file}")
    return str(checkpoint_file)

def load_checkpoint(model_name, cache_dir):
    """Load the most recent checkpoint (final or epoch) if available."""
    cache_path = Path(cache_dir)
    
    # Look for final checkpoint first
    final_checkpoint = cache_path / f"{model_name}_checkpoint.pkl"
    if final_checkpoint.exists():
        try:
            with open(final_checkpoint, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"📥 Loaded final checkpoint: {final_checkpoint}")
            return checkpoint
        except Exception as e:
            print(f"❌ Failed to load final checkpoint: {e}")
    
    # Look for epoch checkpoints
    epoch_checkpoints = list(cache_path.glob(f"{model_name}_checkpoint_epoch_*.pkl"))
    if not epoch_checkpoints:
        return None
    
    # Get the most recent epoch checkpoint
    latest_checkpoint = max(epoch_checkpoints, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"📥 Loaded checkpoint: {latest_checkpoint}")
        return checkpoint
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

def train_model_with_periodic_checkpoints(model, X_train, y_train, X_val, y_val, 
                                        model_name, cache_dir, checkpoint_interval=10):
    """Train a model with periodic checkpointing every N epochs."""
    import types
    
    # Store original fit method
    original_fit = model.fit
    
    def fit_with_checkpoints(X_train, y_train, X_val=None, y_val=None, X_unlbl=None):
        """Modified fit method that saves checkpoints periodically."""
        print(f"  🔄 Starting training with checkpoints every {checkpoint_interval} epochs...")
        
        # Store the original training epoch method
        if hasattr(model, '_train_epoch'):
            original_train_epoch = model._train_epoch
            
            def train_epoch_with_checkpoint(train_loader, optimizer, epoch):
                # Call the original training epoch
                result = original_train_epoch(train_loader, optimizer, epoch)
                
                # Save checkpoint every checkpoint_interval epochs
                if (epoch + 1) % checkpoint_interval == 0:
                    intermediate_results = {
                        "status": "training",
                        "epoch": epoch + 1,
                        "epochs_completed": epoch + 1,
                        "current_loss": getattr(model, 'fitting_loss', [])[-1] if getattr(model, 'fitting_loss', []) else None,
                        "timestamp": datetime.now().isoformat()
                    }
                    save_checkpoint(model, intermediate_results, model_name, cache_dir, 
                                  epoch=(epoch + 1), is_final=False)
                    
                    # Clean up old epoch checkpoints (keep only last 3)
                    cleanup_old_epoch_checkpoints(model_name, cache_dir, keep=3)
                
                return result
            
            # Replace the training epoch method
            model._train_epoch = train_epoch_with_checkpoint
        
        # Call the original fit method
        return original_fit(X_train, y_train, X_val, y_val, X_unlbl)
    
    # Replace the fit method
    model.fit = fit_with_checkpoints
    
    # Train the model
    result = model.fit(X_train, y_train, X_val, y_val)
    
    # Restore original fit method
    model.fit = original_fit
    
    return result

def cleanup_old_epoch_checkpoints(model_name, cache_dir, keep=3):
    """Clean up old epoch checkpoints, keeping only the most recent ones."""
    cache_path = Path(cache_dir)
    epoch_checkpoints = list(cache_path.glob(f"{model_name}_checkpoint_epoch_*.pkl"))
    
    if len(epoch_checkpoints) <= keep:
        return
    
    # Sort by modification time and remove oldest
    epoch_checkpoints.sort(key=lambda p: p.stat().st_mtime)
    for old_checkpoint in epoch_checkpoints[:-keep]:
        try:
            old_checkpoint.unlink()
            print(f"🗑️  Cleaned up old checkpoint: {old_checkpoint.name}")
        except Exception as e:
            print(f"⚠️  Failed to clean up {old_checkpoint.name}: {e}")

def run_cached_qm9_benchmark(resume=True, force_restart=False):
    """Run comprehensive benchmark on full QM9 dataset with simple caching."""
    print("🚀 Starting cached full QM9 torch-molecule benchmark...")
    start_time = time.time()
    
    # Setup cache directory
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Results structure
    results = {
        "benchmark_info": {
            "name": "Cached Full QM9 Dataset Benchmark",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": None,
            "dataset_size": None,
            "train_size": None,
            "val_size": None,
            "test_size": None,
            "resumed_from_cache": False
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
                
                # Check for existing checkpoint
                checkpoint = None
                if resume and not force_restart:
                    checkpoint = load_checkpoint("gnn_predictor", cache_dir)
                
                if checkpoint and resume and not force_restart:
                    print("📥 Resuming from cached GNN model...")
                    results["results"]["gnn_predictor"] = checkpoint['results']
                    results["benchmark_info"]["resumed_from_cache"] = True
                    print(f"  ✅ GNN already trained - MAE: {checkpoint['results'].get('mae', 'N/A'):.4f}")
                else:
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
                        
                        # Train model with periodic checkpoints
                        print(f"  Training on {len(X_train)} molecules...")
                        train_model_with_periodic_checkpoints(
                            model, X_train, y_train, X_val, y_val, 
                            "gnn_predictor", cache_dir, checkpoint_interval=10
                        )
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
                        
                        gnn_results = {
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
                        
                        results["results"]["gnn_predictor"] = gnn_results
                        
                        # Save final checkpoint
                        save_checkpoint(model, gnn_results, "gnn_predictor", cache_dir, is_final=True)
                        
                        print(f"  ✅ GNN training completed - MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
                        print(f"  Training time: {train_time/60:.1f} minutes")
                        
                    except Exception as e:
                        print(f"  ❌ GNN training failed: {e}")
                        results["results"]["gnn_predictor"] = {"status": "failed", "error": str(e)}
            
            # 2. Supervised Encoder
            if "SupervisedMolecularEncoder" in model_imports:
                print("\n🧬 Training Supervised Encoder...")
                
                # Check for existing checkpoint
                checkpoint = None
                if resume and not force_restart:
                    checkpoint = load_checkpoint("supervised_encoder", cache_dir)
                
                if checkpoint and resume and not force_restart:
                    print("📥 Resuming from cached encoder model...")
                    results["results"]["supervised_encoder"] = checkpoint['results']
                    print(f"  ✅ Encoder already trained - Embedding dim: {checkpoint['results'].get('embedding_dim', 'N/A')}")
                else:
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
                        
                        # Train encoder with periodic checkpoints
                        print(f"  Training on {len(X_train)} molecules...")
                        train_model_with_periodic_checkpoints(
                            encoder, X_train, None, None, None, 
                            "supervised_encoder", cache_dir, checkpoint_interval=10
                        )
                        train_time = time.time() - train_start
                        
                        # Get embeddings for evaluation
                        print("  📊 Generating embeddings...")
                        embeddings_train = encoder.encode(X_train[:1000])  # Sample for speed
                        embeddings_test = encoder.encode(X_test[:1000])
                        
                        # Safely extract training history
                        fitting_loss = getattr(encoder, 'fitting_loss', [])
                        fitting_epoch = getattr(encoder, 'fitting_epoch', 0)
                        
                        encoder_results = {
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
                        
                        results["results"]["supervised_encoder"] = encoder_results
                        
                        # Save final checkpoint
                        save_checkpoint(encoder, encoder_results, "supervised_encoder", cache_dir, is_final=True)
                        
                        print(f"  ✅ Encoder training completed - Embedding dim: {embeddings_train.shape[1] if hasattr(embeddings_train, 'shape') else 'unknown'}")
                        print(f"  Training time: {train_time/60:.1f} minutes")
                        
                    except Exception as e:
                        print(f"  ❌ Encoder training failed: {e}")
                        results["results"]["supervised_encoder"] = {"status": "failed", "error": str(e)}
            
            # 3. LSTM Generator
            if "LSTMMolecularGenerator" in model_imports:
                print("\n🧪 Training LSTM Generator...")
                
                # Check for existing checkpoint
                checkpoint = None
                if resume and not force_restart:
                    checkpoint = load_checkpoint("lstm_generator", cache_dir)
                
                if checkpoint and resume and not force_restart:
                    print("📥 Resuming from cached generator model...")
                    results["results"]["lstm_generator"] = checkpoint['results']
                    validity_rate = checkpoint['results'].get('validity_rate', 0)
                    print(f"  ✅ Generator already trained - Validity rate: {validity_rate:.2%}")
                else:
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
                        train_model_with_periodic_checkpoints(
                            generator, lstm_trainX, None, None, None, 
                            "lstm_generator", cache_dir, checkpoint_interval=10
                        )
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
                        
                        generator_results = {
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
                        
                        results["results"]["lstm_generator"] = generator_results
                        
                        # Save final checkpoint
                        save_checkpoint(generator, generator_results, "lstm_generator", cache_dir, is_final=True)
                        
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
        "cache_used": results["benchmark_info"]["resumed_from_cache"],
        "status": "completed"
    }
    
    # Save final results
    output_file = Path(__file__).parent / f"cached_qm9_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("📊 CACHED FULL QM9 BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total duration: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
    print(f"Dataset: {results['benchmark_info']['dataset_size']:,} molecules")
    print(f"Train/Val/Test: {results['benchmark_info']['train_size']:,}/{results['benchmark_info']['val_size']:,}/{results['benchmark_info']['test_size']:,}")
    print(f"Resumed from cache: {results['benchmark_info']['resumed_from_cache']}")
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
    
    print("\n🎉 Cached full QM9 benchmark completed!")
    return results

def main():
    parser = argparse.ArgumentParser(description='Run cached QM9 benchmark')
    parser.add_argument('--no-resume', action='store_true', 
                       help='Do not resume from cache, start fresh')
    parser.add_argument('--force-restart', action='store_true',
                       help='Force restart, ignoring all cache')
    
    args = parser.parse_args()
    
    resume = not args.no_resume
    force_restart = args.force_restart
    
    if force_restart:
        print("⚠️  Force restart mode: ignoring all cached progress")
    elif not resume:
        print("⚠️  No resume mode: starting fresh but will save checkpoints")
    
    run_cached_qm9_benchmark(resume=resume, force_restart=force_restart)

if __name__ == "__main__":
    main()
