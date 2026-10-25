# TopLoRA

Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation [NeurIPS 2025]

## Environment Setup
Please execute the following command to install the necessary dependencies:
```
pip install -r requirements.txt
```

## Dataset 
Please download the datasets from [LLM_Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), place them in the dataset directory, and then copy this directory to the current project.
```
git clone https://github.com/AGI-Edgerunners/LLM-Adapters
cp LLM-Adapters/ft-training_set/math_10k.json LLM-Adapters/dataset
cp LLM-Adapters/ft-training_set/commonsense_170k.json LLM-Adapters/dataset
cp -r LLM-Adapters/dataset ./
rm -rf LLM-Adapters
```

## Run the Experiment
To run a method, specify its corresponding yaml file from the config directory. For example:
```
CUDA_VISIBLE_DEVICES=0 python train_llm.py -f "config/lora.yaml"
CUDA_VISIBLE_DEVICES=0 python train_llm.py -f "config/toplora.yaml"
```

## Citation
```
@inproceedings{li2025toplora,
  title={Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation},
  author={Li, Shiwei and Luo, Xiandi and Wang, Haozhao and Tang, Xing and Cui, Ziqiang and Liu, Dugang and Li, Yuhua and He, Xiuqiang and Li, Ruixuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

# TopLoRA-RoBERTa (Stage 2)

Fork cá»§a repo gá»‘c `toplora-neurips25` Ä‘á»ƒ tÃ¡i táº¡o 4 dÃ²ng TopLoRA trong Table 1 (TopLoRA, NeurIPS 2025) trÃªn GLUE vá»›i RoBERTa-Base/Large, r=8/16, 3 seed (42/43/44), eval trÃªn validation set. KhÃ´ng tÃ¡i táº¡o LoRA baseline.

## Cáº¥u trÃºc

- `mypeft/toplora.py` â€” cÆ¡ cháº¿ TopLoRA (sao chÃ©p nguyÃªn vÄƒn tá»« repo gá»‘c)
- `utils/dev_parse.py`, `utils/train_utils.py` â€” sao chÃ©p nguyÃªn vÄƒn tá»« repo gá»‘c
- `glue_data.py` â€” load GLUE, tokenize, metrics (cola â†’ matthews, stsb â†’ pearson, cÃ²n láº¡i â†’ accuracy)
- `train_roberta.py` â€” huáº¥n luyá»‡n 1 (task, seed, config)
- `params_check.py` â€” kiá»ƒm tra sá»‘ tham sá»‘ trainable trÆ°á»›c khi training
- `config/` â€” 4 yaml: roberta_base_r8 / r16, roberta_large_r8 / r16

## Cháº¡y

```bash
# 0. Kiá»ƒm tra sá»‘ tham sá»‘ (cáº§n internet, ~2GB/model). Expected: 442368 / 884736 / 1179648 / 2359296
python params_check.py

# 1. Train tá»«ng task/seed (cháº¡y láº·p qua 8 task x 3 seed x 4 config)
python train_roberta.py -f config/roberta_base_r8.yaml --task rte --seed 42
python train_roberta.py -f config/roberta_base_r8.yaml --task mnli --seed 43
```

`--task` vÃ  `--seed` trÃªn CLI ghi Ä‘Ã¨ giÃ¡ trá»‹ trong yaml. Káº¿t quáº£ ghi tÄƒng dáº§n vÃ o `{output_dir}/results.json` vá»›i cáº¥u trÃºc `{"task": {"seed": {"validation": float, "mnli_mm": float?, "trainable_lora_params": int}}}`.

## Phá»¥ thuá»™c

```
pip install peft==0.14.0 transformers==4.49.0 datasets scikit-learn scipy python-box accelerate
```
(torch cÃ³ sáºµn trÃªn Kaggle.)

## Nguá»“n siÃªu tham sá»‘

- Section 4.2: lr 3e-4 (base) / 1e-4 (large), alpha=2r, dropout 0.05, target query+value, batch 32
- Appendix A: epochs â€” RTE/MRPC 10, STS-B/CoLA 5, SST-2/QNLI 2, MNLI/QQP 1

## Assumptions (khÃ´ng nÃªu rÃµ trong paper)

- Eval mÃ´ hÃ¬nh cuá»‘i cÃ¹ng sau training, khÃ´ng checkpoint selection (save_strategy='no').
- max_grad_norm Ä‘á»ƒ máº·c Ä‘á»‹nh 1.0 cá»§a Trainer.
- Eval trÃªn validation/dev set vÃ¬ test set GLUE áº©n label.
- Sá»‘ training step má»—i seed: RTE 86 â†’ QQP 11377.
- RoBERTa dÃ¹ng tÃªn module `query`/`value` (khÃ´ng pháº£i `q_proj`/`v_proj`); `convert_target_modules` cá»§a repo gá»‘c sinh tÃªn sai cho RoBERTa nÃªn khÃ´ng dÃ¹ng â€” target_modules ghi trá»±c tiáº¿p trong yaml.
- MNLI eval thÃªm `validation_mismatched` (lÆ°u key `mnli_mm`).

## Fork notes

RoBERTa/GLUE reproduction code (phase 2) added on top of the original NeurIPS 2025 release. Fork of Leopold1423/toplora-neurips25 maintained by @P-Bao.

