"""CPU pre-flight verification for the RoBERTa TopLoRA fork (runs BEFORE burning Kaggle GPU time).

Two parts, no model-weight downloads, CPU only:
  A. Mechanical smoke test on a tiny random RoBERTa:
     LoraConfig(task_type=SEQ_CLS) -> get_peft_model -> replace_toplora_linear
     -> forward + backward. Catches peft API/shape bugs (e.g. get_peft_model kwargs).
  B. Exact trainable-LoRA param counts for the 4 paper configs, built from configs
     (random weights - counts depend on architecture only, no 2GB download):
       roberta-base  r=8 -> 442560, r=16 -> 885120
       roberta-large r=8 -> 1180032, r=16 -> 2360064
     (Per module: lora_A + lora_B + projector weight = 3*hidden*r, plus the
      nn.RMSNorm elementwise gamma of size r -> rounds to paper's 0.44/0.89/1.18/2.36M.)

Usage:  python verify_cpu.py
Exit 0 = all pass. Requires no downloads (all from_config).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from mypeft.toplora import replace_toplora_linear

EXPECTED = {
    ('roberta-base', 8, 16): 442560,
    ('roberta-base', 16, 32): 885120,
    ('roberta-large', 8, 16): 1180032,
    ('roberta-large', 16, 32): 2360064,
}

BASE_CFG = dict(vocab_size=50265, max_position_embeddings=514, type_vocab_size=1)
LARGE_CFG = dict(vocab_size=50265, max_position_embeddings=514, type_vocab_size=1)


def make_cfg(variant):
    cfg = RobertaConfig(
        **(BASE_CFG if variant == 'roberta-base' else dict(
            vocab_size=50265, max_position_embeddings=514, type_vocab_size=1,
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
            intermediate_size=4096)),
        num_labels=2,
        pad_token_id=1,
    )
    return cfg


def count_lora_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)


def part_a_smoke():
    print('=== Part A: tiny-model smoke test (forward + backward through TopLoRA) ===')
    cfg = RobertaConfig(
        vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=64, type_vocab_size=1,
        num_labels=2, pad_token_id=1,
    )
    model = RobertaForSequenceClassification(cfg)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
        target_modules=['query', 'value'], lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    replace_toplora_linear(model)

    lora_layers = [m for m in model.modules() if hasattr(m, 'lora_lambda')]
    assert lora_layers, 'replace_toplora_linear replaced nothing'
    for m in lora_layers:
        w = m.lora_lambda['default'].weight
        assert w.requires_grad, 'lora_lambda weight not trainable'
    n_lambdas = len(lora_layers)
    print(f'  TopLinear layers wrapped: {n_lambdas} (expect 2 layers x 2 modules = 4)')
    assert n_lambdas == 4, n_lambdas

    model.train()
    input_ids = torch.randint(4, 1000, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor([0, 1]))
    loss = out.loss
    assert torch.isfinite(loss), f'loss not finite: {loss}'
    loss.backward()
    grads_ok = sum(
        1 for m in lora_layers if m.lora_lambda['default'].weight.grad is not None
        and torch.isfinite(m.lora_lambda['default'].weight.grad).all()
    )
    print(f'  loss={loss.item():.4f} finite | lora_lambda grads non-None+finite: {grads_ok}/{n_lambdas}')
    assert grads_ok == n_lambdas, 'missing/NaN grads on lora_lambda'
    print('  Part A PASS')
    return True


def part_b_params():
    print('=== Part B: exact trainable-LoRA param counts (from_config, no downloads) ===')
    failed = False
    for (name, rank, alpha), expected in EXPECTED.items():
        model = RobertaForSequenceClassification(make_cfg(name))
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha,
            target_modules=['query', 'value'], lora_dropout=0.05,
        )
        model = get_peft_model(model, peft_config)
        replace_toplora_linear(model)
        actual = count_lora_params(model)
        ok = actual == expected
        failed |= not ok
        print(f'  {name:14s} r={rank:2d} alpha={alpha:2d} | got {actual:9d} | expected {expected:9d} | {"OK" if ok else "MISMATCH"}')
        del model
    print('  Part B PASS' if not failed else '  Part B FAIL')
    return not failed


if __name__ == '__main__':
    torch.manual_seed(0)
    a = part_a_smoke()
    b = part_b_params()
    if a and b:
        print('ALL CHECKS PASSED - safe to run on Kaggle.')
        sys.exit(0)
    print('CHECKS FAILED - do not start Kaggle training yet.')
    sys.exit(1)
