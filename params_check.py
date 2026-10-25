"""Check exact trainable LoRA parameter counts for TopLoRA RoBERTa before training.

Needs internet (downloads ~2GB model weights). Run on Kaggle.

Expected exact counts (paper Table 1: 0.44M / 0.89M / 1.18M / 2.36M):
    roberta-base   r=8  alpha=16 -> 442560
    roberta-base   r=16 alpha=32 -> 885120
    roberta-large  r=8  alpha=16 -> 1180032
    roberta-large  r=16 alpha=32 -> 2360064
  (lora_A + lora_B + projector = 3*hidden*r per module, plus the nn.RMSNorm
   elementwise gamma of size r per module; rounds to the paper's reported values.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from mypeft.toplora import replace_toplora_linear

EXPECTED = {
    ('roberta-base', 8): 442560,
    ('roberta-base', 16): 885120,
    ('roberta-large', 8): 1180032,
    ('roberta-large', 16): 2360064,
}


def count_lora_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)


def build(model_name, rank):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=2 * rank,
        target_modules=['query', 'value'],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    replace_toplora_linear(model)
    count = count_lora_params(model)
    del model
    torch.cuda.empty_cache()
    return count


def main():
    rows = []
    failed = False
    for (model_name, rank), expected in EXPECTED.items():
        actual = build(model_name, rank)
        ok = actual == expected
        status = 'OK' if ok else 'MISMATCH'
        if not ok:
            failed = True
        paper_m = expected / 1e6
        print(f"{model_name:14s} r={rank:2d} alpha={2 * rank:2d} | got {actual:9d} | expected {expected:9d} | paper {paper_m:.2f}M | {status}")
        rows.append((model_name, rank, actual, expected, ok))

    print()
    print(f"{'Model':14s} {'r':>3s} {'#Params':>10s} {'Paper':>8s}")
    for model_name, rank, actual, expected, _ in rows:
        print(f"{model_name:14s} {rank:3d} {actual:10d} {expected / 1e6:7.2f}M")

    if failed:
        print('ERROR: trainable LoRA parameter count mismatch - aborting.')
        sys.exit(1)
    print('All parameter counts match.')


if __name__ == '__main__':
    main()
