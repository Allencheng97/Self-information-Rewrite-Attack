import os
import gc
import json
import torch
import numpy as np
import transformers
from tqdm import tqdm
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from visualize.font_settings import FontSettings
from visualize.visualizer import ContinuousVisualizer
from visualize.legend_settings import ContinuousLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForContinuousVisualization
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
algorithms = ['UPV', 'EWD']
opt_path = "/mnt/datadisk/syj/models/opt-1.3b"
for algorithm in algorithms:
    transformers_config = TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained(opt_path).to(device),
        device=device,
        tokenizer=AutoTokenizer.from_pretrained(opt_path),
        vocab_size=50272,
        max_new_tokens=200,
        min_length=230,
        do_sample=True,
        no_repeat_ngram_size=4
    )

    myWatermark = AutoWatermark.load(
        algorithm,
        algorithm_config=f'config/{algorithm}.json',
        transformers_config=transformers_config
    )

    input_path = '/mnt/datadisk/syj/MarkLLM/opengen/trimmed_opengen_500.jsonl'
    output_path = f'/mnt/datadisk/syj/MarkLLM/opengen/{algorithm}_response_trim_opengen_500_opt27b.json'

    with open(input_path, 'r') as f:
        lines = f.readlines()

    # 获取已写入的行数
    if os.path.exists(output_path):
        with open(output_path, 'r') as out_f:
            existing_lines = sum(1 for _ in out_f)
    else:
        existing_lines = 0

    with open(output_path, 'a') as out_f:  # 追加模式
        for i, line in enumerate(tqdm(lines, desc=f"Processing {algorithm}", unit="line")):
            if i < existing_lines:
                continue

            item = json.loads(line)
            prompt = item['prefix']
            watermarked_text = myWatermark.generate_watermarked_text(prompt)
            unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

            response_item = {
                'prompt': prompt,
                'watermarked_text': watermarked_text,
                'unwatermarked_text': unwatermarked_text
            }
            out_f.write(json.dumps(response_item) + '\n')

    # 清理显存
    del myWatermark, transformers_config, lines, out_f, item, response_item
    gc.collect()
    torch.cuda.empty_cache()