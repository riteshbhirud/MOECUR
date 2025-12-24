#!/usr/bin/env python3

import os
import sys
import json
import copy
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from component.cur_deepseek import is_deepseek_moe_layer, get_deepseek_moe_config

from relaxed_cur import (
    RelaxedCURCompressor,
    RelaxedCURFineTuner,
    get_calibration_loader,
    evaluate_perplexity,
    count_model_storage,
    compute_expected_compression,
)


def apply_relaxed_cur_compression(
    model: nn.Module,
    selected_layers: List[int],
    rank: int,
    cur_mode: str,
    n_iterations: int,
    lr: float,
    lambda_reg: float,
    device: str,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    config = model.config
    moe_config = get_deepseek_moe_config(config)
    
    compressor = RelaxedCURCompressor(
        rank=rank,
        cur_mode=cur_mode,
        n_iterations=n_iterations,
        lr=lr,
        lambda_reg=lambda_reg,
    )
    
    compression_log = {}
    total_original_elements = 0
    total_compressed_elements = 0
    
    layers = model.model.layers
    
    print(f"\nApplying Relaxed CUR compression...")
    print(f"  Rank: {rank}")
    print(f"  Optimization iterations: {n_iterations}")
    print(f"  Lambda (regularization): {lambda_reg}")
    
    for layer_idx in tqdm(selected_layers, desc="Compressing layers"):
        layer = layers[layer_idx]
        
        if not is_deepseek_moe_layer(layer):
            continue
        
        moe = layer.mlp
        layer_device = next(layer.parameters()).device
        
        print(f"\n  Layer {layer_idx}:")
        
        compressed_moe, layer_stats = compressor.compress_moe_layer(
            moe, moe_config, device=layer_device, verbose=verbose
        )
        
        layer.mlp = compressed_moe
        
        total_original_elements += layer_stats['summary']['original_elements']
        total_compressed_elements += layer_stats['summary']['compressed_elements']
        
        summary = layer_stats['summary']
        print(f"    Avg CUR error:     {summary['avg_initial_error']:.4f}")
        print(f"    Avg Relaxed error: {summary['avg_final_error']:.4f}")
        print(f"    Avg reduction:     {summary['avg_error_reduction']:.4f}")
        print(f"    Layer compression: {summary['original_elements']:,} → "
              f"{summary['compressed_elements']:,} "
              f"({summary['space_saving']*100:.1f}% saved)")
        
        compression_log[layer_idx] = layer_stats
        torch.cuda.empty_cache()
    
    storage_stats = {
        'compressed_layers_original': total_original_elements,
        'compressed_layers_new': total_compressed_elements,
        'compressed_layers_saving': 1 - total_compressed_elements / total_original_elements,
    }
    
    return compression_log, storage_stats


def main():
    parser = argparse.ArgumentParser(description='Relaxed CUR for MoE Compression')
    
    parser.add_argument('--model', type=str, default='deepseek-ai/deepseek-moe-16b-base')
    parser.add_argument('--selected_layers', type=str, default='1,2,3,4,5')
    
    parser.add_argument('--rank', type=int, default=512)
    parser.add_argument('--cur_mode', type=str, default='deim')
    parser.add_argument('--n_iterations', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda_reg', type=float, default=0.1)
    
    parser.add_argument('--finetune_steps', type=int, default=300)
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--distill_weight', type=float, default=0.5)
    parser.add_argument('--skip_finetune', action='store_true')
    
    parser.add_argument('--calib_nsamples', type=int, default=64)
    parser.add_argument('--calib_seqlen', type=int, default=512)
    
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.selected_layers == 'all':
        selected_layers = list(range(1, 28))
    else:
        selected_layers = [int(x) for x in args.selected_layers.split(',')]
    
    print("=" * 70)
    print("Relaxed CUR: Bridging Interpretability and Accuracy")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Selected layers: {selected_layers}")
    print(f"\nRelaxed CUR config:")
    print(f"  Rank: {args.rank}")
    print(f"  Optimization iterations: {args.n_iterations}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lambda (regularization): {args.lambda_reg}")
    print(f"\nFine-tuning: {'Enabled' if not args.skip_finetune else 'Disabled'}")
    
    m, n = 1408, 2048
    expected = compute_expected_compression((m, n), args.rank)
    print(f"\nExpected per-projection compression: {expected['space_saving']*100:.1f}%")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    baseline_storage = count_model_storage(model)
    print(f"Baseline storage: {baseline_storage['total_storage']:,} elements "
          f"({baseline_storage['total_storage'] * 2 / 1e9:.2f} GB in bfloat16)")
    
    if not args.skip_finetune:
        print("Creating teacher model copy...")
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    
    if args.evaluate:
        print("\nEvaluating baseline...")
        baseline_ppl = evaluate_perplexity(model, tokenizer, max_samples=100)
        print(f"Baseline PPL: {baseline_ppl:.2f}")
    
    print("\nLoading calibration data...")
    calib_data = get_calibration_loader(
        tokenizer, args.calib_nsamples, args.calib_seqlen, args.seed
    )
    
    print("\n" + "=" * 60)
    print("STAGE 1: Relaxed CUR Compression")
    print("=" * 60)
    
    compression_log, storage_stats = apply_relaxed_cur_compression(
        model=model,
        selected_layers=selected_layers,
        rank=args.rank,
        cur_mode=args.cur_mode,
        n_iterations=args.n_iterations,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        device=args.device,
        verbose=args.verbose,
    )
    
    print("\nRestoring model to GPU...")
    if hasattr(model, 'hf_device_map'):
        delattr(model, 'hf_device_map')
    model = model.to(args.device)
    torch.cuda.empty_cache()
    
    compressed_storage = count_model_storage(model)
    
    if args.evaluate:
        print("\nEvaluating after Relaxed CUR compression...")
        compressed_ppl = evaluate_perplexity(model, tokenizer, max_samples=100)
        print(f"Compressed PPL: {compressed_ppl:.2f} ({compressed_ppl/baseline_ppl:.2f}x)")
    
    if not args.skip_finetune:
        print("\n" + "=" * 60)
        print("STAGE 2: Knowledge Distillation Fine-tuning")
        print("=" * 60)
        
        teacher_model = teacher_model.to(args.device)
        
        finetuner = RelaxedCURFineTuner(
            learning_rate=args.finetune_lr,
            num_steps=args.finetune_steps,
            distill_weight=args.distill_weight,
        )
        
        ft_log = finetuner.finetune(model, teacher_model, calib_data, args.device)
        
        del teacher_model
        torch.cuda.empty_cache()
        
        if args.evaluate:
            print("\nEvaluating after fine-tuning...")
            finetuned_ppl = evaluate_perplexity(model, tokenizer, max_samples=100)
            print(f"Fine-tuned PPL: {finetuned_ppl:.2f} ({finetuned_ppl/baseline_ppl:.2f}x)")
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nStorage (REAL compression):")
    print(f"  Baseline:   {baseline_storage['total_storage']:,} elements")
    print(f"  Compressed: {compressed_storage['total_storage']:,} elements")
    actual_reduction = 1 - compressed_storage['total_storage'] / baseline_storage['total_storage']
    print(f"  Reduction:  {actual_reduction*100:.2f}%")
    
    print(f"\n  Compressed layers only:")
    print(f"    Original:   {storage_stats['compressed_layers_original']:,} elements")
    print(f"    Compressed: {storage_stats['compressed_layers_new']:,} elements")
    print(f"    Saving:     {storage_stats['compressed_layers_saving']*100:.1f}%")
    
    if args.evaluate:
        print(f"\nPerplexity:")
        print(f"  Baseline:   {baseline_ppl:.2f}")
        print(f"  Compressed: {compressed_ppl:.2f} ({compressed_ppl/baseline_ppl:.2f}x)")
        if not args.skip_finetune:
            print(f"  Fine-tuned: {finetuned_ppl:.2f} ({finetuned_ppl/baseline_ppl:.2f}x)")
    
    print(f"\nInterpretability (preserved):")
    print(f"  - col_indices: Which input features selected (stored)")
    print(f"  - row_indices: Which output features selected (stored)")
    print(f"  - col_offset_norms: Column reliability scores (stored)")
    print(f"  - row_offset_norms: Row reliability scores (stored)")
    
    if args.save_path:
        print(f"\nSaving to {args.save_path}...")
        os.makedirs(args.save_path, exist_ok=True)
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        
        with open(os.path.join(args.save_path, 'compression_log.json'), 'w') as f:
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            json.dump(convert(compression_log), f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
