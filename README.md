# Relaxed CUR

Bridging Interpretability and Accuracy in Matrix Decomposition for MoE Compression.

## Overview

Standard CUR decomposition selects exact columns and rows from the original matrix, but correlated selections lead to ~72% reconstruction error. Relaxed CUR adds small learned offsets to selected columns/rows, making them "effectively orthogonal" while maintaining interpretability. This achieves ~42% error while preserving CUR's interpretability benefits.

**Key Insight:**
```
Standard CUR:  C = W[:, indices]           → ~72% error
Relaxed CUR:   C = W[:, indices] + ΔC      → ~42% error
```

**Mathematical Formulation:**
```
min ||W - (C + ΔC) @ U @ (R + ΔR)||² + λ(||ΔC||² + ||ΔR||²)
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
python main.py \
    --model deepseek-ai/deepseek-moe-16b-base \
    --selected_layers 1,2,3,4,5 \
    --rank 512 \
    --n_iterations 500 \
    --lambda_reg 0.1 \
    --evaluate
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `deepseek-ai/deepseek-moe-16b-base` | Model to compress |
| `--selected_layers` | `1,2,3,4,5` | Layers to compress (comma-separated or 'all') |
| `--rank` | `512` | CUR rank (columns/rows to select) |
| `--cur_mode` | `deim` | CUR selection algorithm |
| `--n_iterations` | `500` | Optimization iterations for offsets |
| `--lr` | `0.01` | Learning rate for offset optimization |
| `--lambda_reg` | `0.1` | Regularization weight (higher = more interpretable) |
| `--finetune_steps` | `300` | Knowledge distillation steps |
| `--finetune_lr` | `1e-5` | Fine-tuning learning rate |
| `--distill_weight` | `0.5` | Distillation loss weight |
| `--skip_finetune` | `False` | Skip fine-tuning stage |
| `--evaluate` | `False` | Evaluate perplexity |
| `--save_path` | `None` | Path to save compressed model |

## Project Structure

```
relaxed-cur/
├── main.py                    # CLI entry point
├── relaxed_cur/
│   ├── __init__.py
│   ├── decomposition.py       # RelaxedCURResult, optimization
│   ├── modules.py             # Projection, Expert, MoE modules
│   ├── compressor.py          # RelaxedCURCompressor
│   ├── finetuner.py           # Knowledge distillation
│   └── utils.py               # Calibration, evaluation utilities
├── requirements.txt
└── setup.py
```

## Interpretability

Relaxed CUR preserves interpretability through:
- `col_indices`: Which input features are most important
- `row_indices`: Which output dimensions are most important
- `col_offset_norms`: How much each column needed adjustment (reliability metric)
- `row_offset_norms`: How much each row needed adjustment (reliability metric)

## Dependencies

- `cur.py`: CUR column/row selection (DEIM algorithm)
- `component/cur_deepseek.py`: DeepSeek MoE layer detection and config
