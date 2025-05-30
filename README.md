# SIRA: Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

This repository contains the official code for **Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite
Attacks** , accepted at **ICML 2025**. SIRA introduces a  black-box attack that removes text watermarks from large language models (LLMs) by strategically rewriting specific tokens, preserving semantics while minimizing watermark detection signals.

### Setting up the environment

- python 3.9
- pytorch
- pip install -r requirements.txt

### Run the attack

Enter scripts folder

#### Step 1: Data preparation (Generate Watermarked Text)
```
python generate_responses.py \
  --algorithms UPV EWD \
  --model_path /mnt/models/opt-2.7b \
  --input_path /mnt/c4/processed_c4.json \
  --output_dir /mnt/c4/watermarked/  \
  --cuda_visible_devices 0 \
  --cuda_device 0
```

#### Step 2: Execute Attack Stage 1,2
```
python pre_attack.py \
  --input_dir /mnt/c4/watermarked/ \
  --result_dir /mnt/c4/results/ \
  --model_path /mnt/models/Meta-Llama-3-8B-Instruct \
  --threshold 30 \
  --gpu 0 \
  --algorithms KGW,UPV,DIP
```
#### Step 3: Execute Attack 
```
python attack.py \
  --model_path /mnt/models/Meta-Llama-3-8B-Instruct \
  --input_dir /mnt/c4/results/ \
  --output_dir /mnt/c4/results/ \
  --gpu 0 \
  --algorithms KGW,UPV,DIP
```

Or you can used the merged script after data preparation:

```bash
python attack_onestep.py \
    --orig_input_dir /path/to/your/c4/watermarked/ \
    --intermediate_dir /path/to/your/intermediate_results/ \
    --final_output_dir /path/to/your/attack_results/ \
    --paraphrase_model_path /path/to/your/paraphrase_and_blanking_model \
    --attack_model_path /path/to/your/attack_model \
    --threshold 30 \
    --gpu "0,1" \
    --algorithms "KGW"
```

### Acknowledgement

This code is based on [MarkLLM](https://github.com/THU-BPM/MarkLLM) . Thanks for their wonderful works.

## Citing

```bibtex
@inproceedings{
cheng2025sira,
title={Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite
Attacks},
author={Yixin Cheng and Hongcheng Guo and Yangming Li and Leonid Sigal},
booktitle={International Conference on Machine Learning},
year={2025},
}
```

## License

SIRA is released under MIT License.
