"""Sequential task/seed queue runner for one config (crash-safe).

Usage:
    python run_queue.py --config config/roberta_base_r8.yaml --seeds "42 43 44"

Skips any (task, seed) whose result is already in {output_dir}/results.json,
so an interrupted session can be resumed by re-running the same command.
"""
import os
import sys
import json
import argparse
import subprocess

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
os.chdir(_here)  # config paths and output_dir in the yamls are relative to the fork root

TASK_ORDER = ['rte', 'mrpc', 'stsb', 'cola', 'sst2', 'qnli', 'mnli', 'qqp']


def load_output_dir(config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg['output_dir']


def has_result(output_dir, task, seed):
    path = os.path.join(output_dir, 'results.json')
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            results = json.load(f)
    except Exception:
        return False
    return task in results and str(seed) in results[task]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Yaml path relative to the fork root')
    ap.add_argument('--seeds', default='42', help='Space-separated seeds, e.g. "42 43 44"')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split()]
    output_dir = load_output_dir(args.config)
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    train_log = os.path.join('/kaggle/working', f'train_{cfg_name}.log')

    for task in TASK_ORDER:
        for seed in seeds:
            if has_result(output_dir, task, seed):
                print(f'[skip] {cfg_name} {task} seed {seed} (already in results.json)', flush=True)
                continue
            cmd = [sys.executable, 'train_roberta.py', '-f', args.config,
                   '--task', task, '--seed', str(seed)]
            print(f'[run ] {cfg_name} {task} seed {seed} -> {train_log}', flush=True)
            with open(train_log, 'a') as log:
                log.write(f'\n===== {cfg_name} {task} seed {seed} =====\n')
                log.flush()
                ret = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            if ret != 0:
                print(f'[fail] {cfg_name} {task} seed {seed} exit={ret} (see {train_log})', flush=True)
            else:
                print(f'[done] {cfg_name} {task} seed {seed}', flush=True)
    print(f'[queue finished] {cfg_name}', flush=True)


if __name__ == '__main__':
    main()
