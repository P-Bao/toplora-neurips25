"""GLUE data loading, tokenization and metrics for RoBERTa reproduction of TopLoRA (NeurIPS 2025) Table 1.

Note: MNLI has an extra `validation_mismatched` split - handled in train_roberta.py, not here.
"""
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr

# Per-task training epochs (paper Appendix A)
TASK_EPOCHS = {
    'rte': 10,
    'mrpc': 10,
    'stsb': 5,
    'cola': 5,
    'sst2': 2,
    'qnli': 2,
    'mnli': 1,
    'qqp': 1,
}

# Text column keys in HF datasets.load_dataset('glue', task)
TASK_KEYS = {
    'cola': ('sentence', None),
    'sst2': ('sentence', None),
    'mrpc': ('sentence1', 'sentence2'),
    'rte': ('sentence1', 'sentence2'),
    'stsb': ('sentence1', 'sentence2'),
    'qqp': ('question1', 'question2'),
    'qnli': ('question', 'sentence'),
    'mnli': ('premise', 'hypothesis'),
}

NUM_LABELS = {t: 2 for t in TASK_KEYS}
NUM_LABELS['stsb'] = 1
NUM_LABELS['mnli'] = 3


def tokenize_task(tokenizer, task, max_length):
    """Load GLUE task and tokenize train/validation splits.

    Dynamic padding: padding=False here, DataCollatorWithPadding pads per batch.
    Label column kept as-is; original text columns removed.
    """
    key1, key2 = TASK_KEYS[task]
    ds = load_dataset('glue', task)
    tokenized = ds.map(
        lambda ex: tokenizer(
            ex[key1],
            ex[key2] if key2 is not None else None,
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        batched=True,
        remove_columns=[c for c in ds['train'].column_names if c != 'label'],
    )
    return tokenized


def compute_metrics_fn(task):
    """Return a compute_metrics callable for transformers.Trainer."""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task == 'stsb':
            preds = predictions[:, 0]
            labels = labels.astype('float32')
            return {'pearson': float(pearsonr(preds, labels)[0])}
        preds = predictions.argmax(axis=-1)
        if task == 'cola':
            return {'matthews_correlation': float(matthews_corrcoef(labels, preds))}
        return {'accuracy': float(accuracy_score(labels, preds))}
    return compute_metrics
