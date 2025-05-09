# SIRA: Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

This repository contains the official code for **Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite
Attacks** , accepted at **ICML 2025**. SIRA introduces a  black-box attack that removes text watermarks from large language models (LLMs) by strategically rewriting specific tokens, preserving semantics while minimizing watermark detection signals.

#### Setting up the environment

- python 3.9
- pytorch
- pip install -r requirements.txt

#### Run the attack

#### Acknowledgement
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
