"""Fine-tune RoBERTa with TopLoRA on a GLUE task (TopLoRA NeurIPS 2025 Table 1 reproduction, stage 2).

Usage:
    python train_roberta.py -f config/roberta_base_r8.yaml --task rte --seed 42

`--task` and `--seed` CLI values override any value in the yaml.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

from utils.dev_parse import load_config, print_dict_paths
from utils.train_utils import set_global_seed
from glue_data import TASK_EPOCHS, NUM_LABELS, tokenize_task, compute_metrics_fn


def parse_task_seed():
    """Parse --task/--seed directly, then strip them from sys.argv.

    load_config() re-parses sys.argv and only accepts `--key=value` overrides
    that match existing yaml keys, so `--task rte` would crash it.
    """
    parser = argparse.ArgumentParser(description='RoBERTa TopLoRA GLUE training')
    parser.add_argument('--task', type=str, required=True, choices=sorted(TASK_EPOCHS.keys()))
    parser.add_argument('--seed', type=int, default=None)
    args, _ = parser.parse_known_args()
    cleaned = []
    skip = False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a.split('=')[0] in ('--task', '--seed'):
            if '=' not in a:
                skip = True
            continue
        cleaned.append(a)
    sys.argv = [sys.argv[0]] + cleaned
    return args


def count_lora_params(model):
    """Count trainable parameters whose name contains 'lora_'."""
    return sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)


def main():
    args = parse_task_seed()
    _, config = load_config()

    task = args.task
    # CLI overrides yaml values
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    config['seed'] = seed

    print(f"Task: {task} | Seed: {seed} | Model: {config.model}")
    print_dict_paths(config)

    set_global_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    # Pure FP32 for GLUE (paper setting): no torch_dtype passed.
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=NUM_LABELS[task]
    )

    lora_cfg = config.lora_config
    peft_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
    )
    model = get_peft_model(model, peft_config, task_type=TaskType.SEQ_CLS)
    if lora_cfg.get('use_toplora', False):
        from mypeft.toplora import replace_toplora_linear
        replace_toplora_linear(model)

    trainable_lora_params = count_lora_params(model)
    print(f"Trainable LoRA params (names containing 'lora_'): {trainable_lora_params}")

    train_cfg = config.train_config
    tokenized = tokenize_task(tokenizer, task, train_cfg.max_length)
    train_ds = tokenized['train']
    eval_datasets = {'validation': tokenized['validation']}
    if task == 'mnli':
        eval_datasets['validation_mm'] = tokenized['validation_mismatched']

    accum = max(1, train_cfg.batch_size // train_cfg.micro_batch_size)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=train_cfg.lr,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type='linear',
        per_device_train_batch_size=train_cfg.micro_batch_size,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=train_cfg.get('gradient_checkpointing', False),
        num_train_epochs=TASK_EPOCHS[task],
        weight_decay=0.0,
        logging_steps=50,
        save_strategy='no',
        report_to='none',
        seed=seed,
        # GLUE runs in FP32 - do not enable fp16/bf16
        per_device_eval_batch_size=128,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_datasets['validation'],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn(task),
    )

    train_result = trainer.train()
    print(f"Training done: {train_result.metrics}")

    # Evaluate final model (no checkpoint selection - assumption, see README)
    metric_name = {
        'cola': 'matthews_correlation',
        'stsb': 'pearson',
    }.get(task, 'accuracy')

    os.makedirs(config.output_dir, exist_ok=True)
    results_path = os.path.join(config.output_dir, 'results.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)

    task_results = {}
    for name, dataset in eval_datasets.items():
        metrics = trainer.evaluate(eval_dataset=dataset)
        value = metrics[f'eval_{metric_name}']
        task_results['mnli_mm' if name == 'validation_mm' else 'validation'] = value
        print(f"{name}: {metric_name} = {value:.4f}")
    task_results['trainable_lora_params'] = trainable_lora_params

    task_store = results.setdefault(task, {})
    task_store[str(seed)] = task_results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written (crash-safe incrementally) to {results_path}")

    print('===== SUMMARY =====')
    for t, seeds in results.items():
        for s, res in seeds.items():
            metric_str = ', '.join(f"{k}={v:.4f}" for k, v in res.items() if k != 'trainable_lora_params')
            print(f"{t} | seed {s} | {metric_str}")


if __name__ == '__main__':
    main()
