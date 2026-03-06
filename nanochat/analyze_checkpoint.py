#!/usr/bin/env python3
"""
Analyze a nanochat model checkpoint and display detailed information.

This script reads a checkpoint directory and displays:
- Model architecture details (layers, heads, embedding size)
- Total parameter count
- Checkpoint file sizes
- Training configuration

Usage:
    python analyze_checkpoint.py <checkpoint_dir>
    python analyze_checkpoint.py cache/base_checkpoints/d20
    python analyze_checkpoint.py --checkpoint cache/mid_checkpoints/d24/model_003500.pt
    python analyze_checkpoint.py --latest base  # Analyze latest base checkpoint
    python analyze_checkpoint.py --latest mid   # Analyze latest mid checkpoint
    python analyze_checkpoint.py --latest sft   # Analyze latest sft checkpoint
    python analyze_checkpoint.py --latest rl    # Analyze latest rl checkpoint

Author: Jason Cox
Date: 2026-01-19
"""

import argparse
import json
import os
import torch
from pathlib import Path


def human_readable_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def count_parameters(state_dict):
    """Count total parameters in a model state dict."""
    total_params = 0
    param_details = {}
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            param_count = param.numel()
            total_params += param_count
            param_details[name] = {
                'shape': list(param.shape),
                'params': param_count,
                'dtype': str(param.dtype)
            }
    
    return total_params, param_details


def find_checkpoint_files(checkpoint_path):
    """
    Find checkpoint files given a path that could be:
    - A directory containing checkpoints
    - A specific model_*.pt file
    - A meta_*.json file
    """
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.is_file():
        # If it's a file, determine if it's a model or meta file
        if checkpoint_path.name.startswith('model_'):
            model_file = checkpoint_path
            meta_file = checkpoint_path.parent / checkpoint_path.name.replace('model_', 'meta_').replace('.pt', '.json')
        elif checkpoint_path.name.startswith('meta_'):
            meta_file = checkpoint_path
            model_file = checkpoint_path.parent / checkpoint_path.name.replace('meta_', 'model_').replace('.json', '.pt')
        else:
            raise ValueError(f"Unknown checkpoint file format: {checkpoint_path}")
    elif checkpoint_path.is_dir():
        # Find the latest checkpoint in the directory
        model_files = sorted(checkpoint_path.glob('model_*.pt'))
        if not model_files:
            raise ValueError(f"No model checkpoints found in {checkpoint_path}")
        model_file = model_files[-1]
        meta_file = checkpoint_path / model_file.name.replace('model_', 'meta_').replace('.pt', '.json')
    else:
        raise ValueError(f"Path does not exist: {checkpoint_path}")
    
    return model_file, meta_file


def find_latest_checkpoint(checkpoint_type):
    """Find the latest checkpoint of a given type (base, mid, sft, rl)."""
    base_dir = Path.home() / '.cache' / 'nanochat'
    
    type_map = {
        'base': 'base_checkpoints',
        'mid': 'mid_checkpoints',
        'sft': 'sft_checkpoints',
        'rl': 'rl_checkpoints'
    }
    
    if checkpoint_type not in type_map:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}. Use: base, mid, sft, or rl")
    
    checkpoint_dir = base_dir / type_map[checkpoint_type]
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find all depth directories (d20, d24, etc.)
    depth_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('d')])
    
    if not depth_dirs:
        raise ValueError(f"No depth directories found in {checkpoint_dir}")
    
    # Use the last depth directory (typically the one being trained)
    latest_depth_dir = depth_dirs[-1]
    
    # Find the latest checkpoint in that directory
    model_files = sorted(latest_depth_dir.glob('model_*.pt'))
    if not model_files:
        raise ValueError(f"No model checkpoints found in {latest_depth_dir}")
    
    return model_files[-1].parent


def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint and return detailed information."""
    # Find checkpoint files
    model_file, meta_file = find_checkpoint_files(checkpoint_path)
    
    # Check if files exist
    if not model_file.exists():
        raise ValueError(f"Model file not found: {model_file}")
    if not meta_file.exists():
        raise ValueError(f"Meta file not found: {meta_file}")
    
    # Read metadata
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    
    model_config = meta.get('model_config', {})
    user_config = meta.get('user_config', {})
    
    # Load model checkpoint
    print(f"Loading checkpoint: {model_file}")
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=True)
    
    # Count parameters
    total_params, param_details = count_parameters(checkpoint)
    
    # Get file sizes
    model_size = model_file.stat().st_size
    meta_size = meta_file.stat().st_size
    optim_file = model_file.parent / model_file.name.replace('model_', 'optim_')
    optim_size = optim_file.stat().st_size if optim_file.exists() else 0
    
    return {
        'model_file': model_file,
        'meta_file': meta_file,
        'model_config': model_config,
        'user_config': user_config,
        'step': meta.get('step', 'unknown'),
        'val_bpb': meta.get('val_bpb', 'unknown'),
        'total_params': total_params,
        'param_details': param_details,
        'model_size': model_size,
        'meta_size': meta_size,
        'optim_size': optim_size,
        'total_size': model_size + meta_size + optim_size
    }


def format_number(num):
    """Format large numbers with commas and suffixes."""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return str(num)


def print_checkpoint_info(info):
    """Print formatted checkpoint information."""
    print("\n" + "="*70)
    print("NANOCHAT MODEL CHECKPOINT ANALYSIS")
    print("="*70)
    
    print(f"\n📁 Checkpoint Location:")
    print(f"   Model:  {info['model_file']}")
    print(f"   Meta:   {info['meta_file']}")
    
    print(f"\n📊 Training Progress:")
    print(f"   Step:       {info['step']:,}")
    if info['val_bpb'] != 'unknown':
        print(f"   Val BPB:    {info['val_bpb']:.4f}")
    
    config = info['model_config']
    print(f"\n🏗️  Model Architecture:")
    print(f"   Layers:           {config.get('n_layer', 'unknown')}")
    print(f"   Attention Heads:  {config.get('n_head', 'unknown')}")
    print(f"   KV Heads:         {config.get('n_kv_head', 'unknown')}")
    print(f"   Embedding Size:   {config.get('n_embd', 'unknown')}")
    print(f"   Vocabulary Size:  {config.get('vocab_size', 'unknown'):,}")
    print(f"   Sequence Length:  {config.get('sequence_len', 'unknown'):,}")
    
    print(f"\n🔢 Parameters:")
    print(f"   Total Parameters: {info['total_params']:,} ({format_number(info['total_params'])})")
    
    # Calculate params per layer
    if config.get('n_layer'):
        params_per_layer = info['total_params'] / config['n_layer']
        print(f"   Per Layer:        ~{format_number(params_per_layer)}")
    
    print(f"\n💾 Storage:")
    print(f"   Model File:       {human_readable_size(info['model_size'])}")
    print(f"   Meta File:        {human_readable_size(info['meta_size'])}")
    if info['optim_size'] > 0:
        print(f"   Optimizer File:   {human_readable_size(info['optim_size'])}")
    print(f"   Total Size:       {human_readable_size(info['total_size'])}")
    
    user_config = info['user_config']
    if user_config:
        print(f"\n⚙️  Training Configuration:")
        if 'run' in user_config:
            print(f"   Run Name:         {user_config['run']}")
        if 'device_batch_size' in user_config:
            print(f"   Device Batch:     {user_config['device_batch_size']}")
        if 'total_batch_size' in user_config:
            print(f"   Total Batch:      {user_config['total_batch_size']:,}")
        if 'matrix_lr' in user_config:
            print(f"   Matrix LR:        {user_config['matrix_lr']}")
        if 'grad_clip' in user_config:
            print(f"   Gradient Clip:    {user_config['grad_clip']}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze nanochat model checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a specific checkpoint directory
  python analyze_checkpoint.py cache/base_checkpoints/d20
  
  # Analyze a specific checkpoint file
  python analyze_checkpoint.py cache/base_checkpoints/d20/model_021400.pt
  
  # Analyze latest checkpoint of each type
  python analyze_checkpoint.py --latest base
  python analyze_checkpoint.py --latest mid
  python analyze_checkpoint.py --latest sft
  python analyze_checkpoint.py --latest rl
  
  # Show detailed parameter breakdown
  python analyze_checkpoint.py cache/base_checkpoints/d20 --verbose
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('checkpoint_path', nargs='?',
                      help='Path to checkpoint directory or file')
    group.add_argument('--latest', choices=['base', 'mid', 'sft', 'rl'],
                      help='Analyze latest checkpoint of given type')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Show detailed parameter breakdown')
    
    args = parser.parse_args()
    
    try:
        # Determine checkpoint path
        if args.latest:
            checkpoint_path = find_latest_checkpoint(args.latest)
            print(f"Using latest {args.latest} checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = args.checkpoint_path
        
        # Analyze checkpoint
        info = analyze_checkpoint(checkpoint_path)
        
        # Print results
        print_checkpoint_info(info)
        
        # Verbose output
        if args.verbose:
            print("📋 Detailed Parameter Breakdown:")
            print("-" * 70)
            for name, details in sorted(info['param_details'].items()):
                shape_str = "x".join(str(s) for s in details['shape'])
                print(f"   {name:50s} {shape_str:20s} {format_number(details['params']):>10s}")
            print("-" * 70)
            print(f"   {'TOTAL':50s} {' ':20s} {format_number(info['total_params']):>10s}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
