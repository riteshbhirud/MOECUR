from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from .modules import RelaxedCURMoE


def get_calibration_loader(tokenizer, nsamples=64, seqlen=512, seed=42):
    data = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    samples = []
    for i, example in enumerate(data):
        if len(samples) >= nsamples * 2:
            break
        if len(example['text']) > 100:
            samples.append(example['text'])
    
    encodings = []
    for text in samples:
        if len(encodings) >= nsamples:
            break
        tokens = tokenizer(
            text, truncation=True, max_length=seqlen,
            padding='max_length', return_tensors='pt'
        )
        if tokens['input_ids'].shape[1] == seqlen:
            encodings.append({
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask']
            })
    
    return encodings


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, max_samples=None):
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(data['text'])
    
    if max_samples:
        text = text[:max_samples * 100]
    
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=50000)
    
    model.eval()
    nlls = []
    seq_len = 512
    device = next(model.parameters()).device
    
    for begin_loc in tqdm(range(0, min(encodings.input_ids.size(1), 8192), seq_len),
                          desc="Evaluating PPL"):
        end_loc = min(begin_loc + seq_len, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        
        outputs = model(input_ids, labels=input_ids, use_cache=False)
        nlls.append(outputs.loss * (end_loc - begin_loc))
    
    ppl = torch.exp(torch.stack(nlls).sum() / min(encodings.input_ids.size(1), 8192))
    return ppl.item()


def count_model_storage(model, compressed_layers: Dict[int, RelaxedCURMoE] = None) -> Dict:
    total_params = 0
    total_buffers = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
    
    for name, buf in model.named_buffers():
        total_buffers += buf.numel()
    
    return {
        'parameters': total_params,
        'buffers': total_buffers,
        'total_storage': total_params + total_buffers,
    }


def compute_expected_compression(original_shape: Tuple[int, int], rank: int) -> Dict:
    m, n = original_shape
    k = rank
    
    original_elements = m * n
    compressed_elements = m * k + k * k + k * n + 2 * k + 2 * k
    
    return {
        'original_elements': original_elements,
        'compressed_elements': compressed_elements,
        'compression_ratio': compressed_elements / original_elements,
        'space_saving': 1 - compressed_elements / original_elements,
    }
