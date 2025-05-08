# SIRA: Self-Information Rewrite Attack on LLM Watermarks

This repository contains the official code for **SIRA** (Self-Information Rewrite Attack), accepted at **ICML 2025**. SIRA introduces a  black-box attack that removes text watermarks from large language models (LLMs) by strategically rewriting high-entropy tokens, preserving semantics while minimizing watermark detection signals.

#### Setting up the environment

- python 3.9
- pytorch
- pip install -r requirements.txt

*Tips:* If you wish to utilize the EXPEdit or ITSEdit algorithm, you will need to import for .pyx file, take EXPEdit as an example:

- run `python watermark/exp_edit/cython_files/setup.py build_ext --inplace`
- move the generated `.so` file into `watermark/exp_edit/cython_files/`

### Acknowledgement
This code is based on [MarkLLM](https://github.com/THU-BPM/MarkLLM) . Thanks for their wonderful works.
